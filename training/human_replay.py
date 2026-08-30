"""Turn a persisted browser replay into an expert-policy JSONL shard.

The browser log contains partial action sessions, cancelled attempts, AI moves,
and explicit turn-boundary events.  This module replays the complete event
stream through the Rust engine, then keeps only confirmed, non-undone actions
for the requested human seat.

The Rust extension is intentionally imported at runtime.  The normal training
container supplies the Linux extension, while unit tests can exercise the
pure event/choice helpers without importing it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model_self_play import detect_engine_revision
from .schema import SELF_PLAY_FORMAT, SELF_PLAY_FORMAT_VERSION


ROOT_ACTIONS = {
    "buildbuilding": 0,
    "buildbuildingaction": 0,
    "buildrailroad": 1,
    # The UI starts a BuildRailroad session; the engine records the resolved
    # two-link variant as BuildDoubleRailroad in ReplayMove.
    "builddoublerailroad": 1,
    "develop": 2,
    "developdouble": 3,
    "sell": 4,
    "loan": 5,
    "scout": 6,
    "pass": 7,
}
INDUSTRIES = {
    "coal": 0,
    "iron": 1,
    "beer": 2,
    "goods": 3,
    "pottery": 4,
    "cotton": 5,
}
N_BUILD_LOCATIONS = 49
MARKET_SOURCE = N_BUILD_LOCATIONS
N_TRADE_POST_SLOTS = 12


@dataclass
class ConfirmedAction:
    action_type: str
    choices: list[tuple[str, Any]]
    state_record: dict
    legal_record: dict
    before_save: Any
    position_key: str | None
    replay_move: dict | None = None
    undone: bool = False


@dataclass(frozen=True)
class ReplayIntegrityAudit:
    """Structural audit of a browser event stream before training it."""

    replay_integrity: str
    value_target_usable: bool
    anomaly_counts: dict[str, int]
    anomaly_events: dict[str, tuple[int, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "replay_integrity": self.replay_integrity,
            "value_target_usable": self.value_target_usable,
            "anomaly_counts": dict(self.anomaly_counts),
            "anomaly_events": {
                name: list(indices) for name, indices in self.anomaly_events.items()
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export confirmed human browser actions as an expert JSONL shard."
    )
    parser.add_argument("--database", default="games.sqlite3")
    parser.add_argument("--game-id", type=int, default=6)
    parser.add_argument("--human-player", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=24.0,
        help="Per-position policy-loss multiplier stored in the expert shard",
    )
    parser.add_argument("--engine-revision")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        not math.isfinite(args.policy_loss_weight)
        or args.policy_loss_weight <= 0.0
        or args.policy_loss_weight > 256.0
    ):
        raise ValueError("--policy-loss-weight must be finite and between 0 and 256")
    if args.game_id < 0 or args.human_player < 0:
        raise ValueError("--game-id and --human-player must be non-negative")
    export_human_replay(
        database=Path(args.database),
        game_id=args.game_id,
        human_player=args.human_player,
        output=Path(args.output),
        policy_loss_weight=args.policy_loss_weight,
        engine_revision=args.engine_revision or detect_engine_revision(),
        overwrite=args.overwrite,
    )


def export_human_replay(
    *,
    database: str | Path,
    game_id: int,
    human_player: int,
    output: str | Path,
    policy_loss_weight: float = 24.0,
    engine_revision: str | None = None,
    overwrite: bool = False,
    engine_module: Any | None = None,
) -> dict:
    """Replay one database game and write a single-game expert shard."""
    if game_id < 0 or human_player < 0:
        raise ValueError("game_id and human_player must be non-negative")
    if (
        not math.isfinite(float(policy_loss_weight))
        or float(policy_loss_weight) <= 0.0
        or float(policy_loss_weight) > 256.0
    ):
        raise ValueError("policy_loss_weight must be finite and between 0 and 256")

    row = _load_game_row(Path(database), game_id)
    num_players = _non_negative_int(row["num_players"], "num_players")
    seed = _non_negative_int(row["seed"], "seed")
    if human_player >= num_players:
        raise ValueError(
            f"human_player {human_player} is outside the {num_players}-player game"
        )
    events = _decode_events(row["action_log"])
    audit = audit_replay_events(events, num_players=num_players)

    if engine_module is None:
        try:
            import fast_brass as engine_module  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "fast_brass Python extension is required; run this command in the training container"
            ) from error

    game = engine_module.BrassRLGame(num_players, seed)
    schema = game.get_training_feature_schema()
    records = _replay_events(game, events)
    if not game.is_done():
        raise RuntimeError("event replay ended before the game reached a terminal state")
    outcome = game.get_outcome()
    positions = _build_expert_positions(
        records,
        human_player=human_player,
        game_index=game_id,
        game_seed=seed,
        outcome=outcome,
        policy_loss_weight=float(policy_loss_weight),
        value_target_usable=audit.value_target_usable,
    )
    if not positions:
        raise RuntimeError("the replay contains no final confirmed actions for the human seat")

    output_path = Path(output).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing shard {output_path}")
    partial = Path(f"{output_path}.partial")
    if partial.exists():
        raise FileExistsError(f"incomplete shard already exists at {partial}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    revision = (engine_revision or detect_engine_revision()).strip()
    if not revision:
        raise ValueError("engine_revision must not be empty")

    victory_points = [int(value) for value in outcome["victory_points"]]
    header = {
        "record_type": "metadata",
        "format": SELF_PLAY_FORMAT,
        "format_version": SELF_PLAY_FORMAT_VERSION,
        "crate_version": _installed_crate_version(),
        "engine_revision": revision,
        "feature_schema": schema,
        "games": 1,
        "game_index_offset": game_id,
        "num_players": num_players,
        "source_type": "human_replay",
        "source_game_id": game_id,
        "human_player": human_player,
        "human_final_vp": victory_points[human_player],
        "human_final_victory_points": victory_points,
        "expert_positions": len(positions),
        "expert_target": "confirmed_non_undone_action_one_hot",
        "policy_loss_weight": float(policy_loss_weight),
        "replay_integrity": audit.replay_integrity,
        "value_target_usable": audit.value_target_usable,
        "replay_anomaly_counts": audit.anomaly_counts,
        "replay_anomaly_events": audit.anomaly_events,
    }
    game_record = {
        "record_type": "game",
        "format_version": SELF_PLAY_FORMAT_VERSION,
        "game_index": game_id,
        "game_seed": seed,
        "positions": len(positions),
        "official_winners": [int(value) for value in outcome["official_winners"]],
        "shared_win_values": [float(value) for value in outcome["shared_win_values"]],
        "placements": [int(value) for value in outcome["placements"]],
        "finish_order": [int(value) for value in outcome["finish_order"]],
        "victory_points": victory_points,
        "victory_point_margins": [
            int(value) for value in outcome["victory_point_margins"]
        ],
        "income_levels": [int(value) for value in outcome["income_levels"]],
        "money": [int(value) for value in outcome["money"]],
        "replay_integrity": audit.replay_integrity,
        "value_target_usable": audit.value_target_usable,
        "replay_anomaly_counts": audit.anomaly_counts,
    }

    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            _write_json_line(handle, header)
            _write_json_line(handle, game_record)
            for position in positions:
                _write_json_line(handle, position)
            handle.flush()
            os.fsync(handle.fileno())
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output appeared while exporting: {output_path}")
        partial.replace(output_path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    summary = {
        "output": str(output_path),
        "game_id": game_id,
        "human_player": human_player,
        "num_players": num_players,
        "events": len(events),
        "confirmed_actions": len(records),
        "expert_positions": len(positions),
        "victory_points": victory_points,
        "official_winners": game_record["official_winners"],
        "policy_loss_weight": float(policy_loss_weight),
        "replay_integrity": audit.replay_integrity,
        "value_target_usable": audit.value_target_usable,
        "replay_anomaly_counts": dict(audit.anomaly_counts),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _load_game_row(database: Path, game_id: int) -> dict[str, Any]:
    path = database.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, num_players, seed, action_log FROM games WHERE id = ?",
            (game_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"game id {game_id} was not found in {path}")
    return dict(row)


def _decode_events(raw_action_log: Any) -> list[dict]:
    if not isinstance(raw_action_log, str):
        raise ValueError("games.action_log must be text")
    try:
        events = json.loads(raw_action_log)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid action_log JSON: {error}") from error
    if not isinstance(events, list):
        raise ValueError("action_log must contain a JSON array")
    decoded: list[dict] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("kind"), str):
            raise ValueError(f"action_log event {index} must contain a string kind")
        decoded.append(event)
    return decoded


def audit_replay_events(
    events: Iterable[dict], *, num_players: int | None = None
) -> ReplayIntegrityAudit:
    """Audit turn boundaries and replay snapshots without mutating an engine.

    A duplicate ``end_turn`` can advance a second seat before its action is
    recorded.  Such a stream may still be replayable by the legacy engine, but
    its terminal value labels are not trustworthy, so value training is
    enabled only for a structurally clean stream.
    """

    events = list(events)
    counts: Counter[str] = Counter()
    event_indices: defaultdict[str, list[int]] = defaultdict(list)

    def anomaly(name: str, index: int, amount: int = 1) -> None:
        counts[name] += amount
        if len(event_indices[name]) < 32:
            event_indices[name].append(index)

    pending = False
    turn_open = False
    confirmed_since_end = 0
    last_kind: str | None = None
    tokens: list[dict[str, Any]] = []
    unbound: deque[dict[str, Any]] = deque()
    undo_stack: list[dict[str, Any]] = []
    replay_tokens: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            anomaly("malformed_event", event_index)
            last_kind = None
            continue
        kind = event.get("kind")
        if not isinstance(kind, str):
            anomaly("malformed_event", event_index)
            last_kind = None
            continue

        if kind == "start_turn":
            turn_open = True
        elif kind == "start_action":
            if not turn_open:
                anomaly("action_without_start_turn", event_index)
                turn_open = True
            if pending:
                anomaly("overlapping_action_session", event_index)
            pending = True
        elif kind == "apply_choice":
            if not pending:
                anomaly("choice_without_action_session", event_index)
        elif kind == "confirm_action":
            if not pending:
                anomaly("confirm_without_action_session", event_index)
            else:
                token = {
                    "event_index": event_index,
                    "replay_move": None,
                    "undone": False,
                }
                tokens.append(token)
                unbound.append(token)
                undo_stack.append(token)
                confirmed_since_end += 1
                pending = False
        elif kind == "cancel_action":
            pending = False
        elif kind == "undo_last_action":
            while undo_stack and undo_stack[-1]["undone"]:
                undo_stack.pop()
            if not undo_stack:
                anomaly("undo_without_confirmed_action", event_index)
            else:
                token = undo_stack.pop()
                token["undone"] = True
                confirmed_since_end = max(0, confirmed_since_end - 1)
        elif kind == "end_turn":
            if last_kind == "end_turn":
                anomaly("consecutive_end_turn", event_index)
            if not turn_open:
                anomaly("end_turn_without_active_turn", event_index)
            if confirmed_since_end == 0:
                anomaly("end_turn_without_confirmed_action", event_index)
            if pending:
                anomaly("end_turn_with_action_session", event_index)
            pending = False
            turn_open = False
            confirmed_since_end = 0
            undo_stack.clear()
        elif kind == "replay_move":
            if not unbound:
                anomaly("replay_move_without_confirmation", event_index)
            else:
                token = unbound.popleft()
                token["replay_move"] = event
                token["replay_event_index"] = event_index
                replay_tokens.append(token)
                _audit_replay_snapshot(
                    event,
                    event_index=event_index,
                    num_players=num_players,
                    anomaly=anomaly,
                )
        elif kind in {"replay_analysis", "resolve_shortfall"}:
            pass
        else:
            anomaly("unsupported_event", event_index)
        last_kind = kind

    if pending:
        anomaly("unconfirmed_action_session", len(events))
    if unbound:
        anomaly("confirmation_without_replay_move", len(events), len(unbound))

    effective = [
        token
        for token in replay_tokens
        if not token["undone"] and isinstance(token.get("replay_move"), dict)
    ]
    per_turn: Counter[tuple[int, int]] = Counter()
    parsed_positions: list[tuple[dict, tuple[str, int, int, int, int, bool], int]] = []
    for token in effective:
        replay_move = token["replay_move"]
        assert isinstance(replay_move, dict)
        parsed = _parse_position_key(replay_move.get("position_key"))
        if parsed is not None:
            parsed_positions.append((replay_move, parsed, int(token.get("replay_event_index", 0))))
            _, _, turn_count, player_idx, _, _ = parsed
            per_turn[(turn_count, player_idx)] += 1

    for (turn_count, _player_idx), action_count in per_turn.items():
        if action_count > 2:
            anomaly("turn_action_overflow", turn_count, action_count - 2)

    for previous, current in zip(parsed_positions, parsed_positions[1:]):
        _previous_move, previous_key, _previous_event_index = previous
        _current_move, current_key, current_event_index = current
        previous_turn = previous_key[2]
        current_turn = current_key[2]
        previous_player = previous_key[3]
        current_player = current_key[3]
        if current_turn < previous_turn:
            anomaly("turn_count_regression", current_event_index)
        elif current_turn - previous_turn > 1:
            anomaly(
                "skipped_turns",
                current_event_index,
                current_turn - previous_turn - 1,
            )
        if current_turn != previous_turn and current_player == previous_player:
            anomaly("same_player_after_turn_advance", current_event_index)
        elif current_turn == previous_turn and current_player != previous_player:
            anomaly("player_changed_without_turn_advance", current_event_index)

    anomaly_counts = dict(sorted(counts.items()))
    anomaly_events = {
        name: tuple(event_indices[name]) for name in sorted(event_indices)
    }
    clean = not anomaly_counts
    return ReplayIntegrityAudit(
        replay_integrity="clean" if clean else "contaminated",
        value_target_usable=clean,
        anomaly_counts=anomaly_counts,
        anomaly_events=anomaly_events,
    )


def _parse_position_key(
    raw: Any,
) -> tuple[str, int, int, int, int, bool] | None:
    if not isinstance(raw, str):
        return None
    parts = raw.split(":")
    if len(parts) != 6 or parts[5] not in {"true", "false"}:
        return None
    try:
        return (
            parts[0],
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            parts[5] == "true",
        )
    except ValueError:
        return None


def _audit_replay_snapshot(
    replay_move: dict,
    *,
    event_index: int,
    num_players: int | None,
    anomaly: Any,
) -> None:
    parsed = _parse_position_key(replay_move.get("position_key"))
    if parsed is None:
        anomaly("invalid_position_key", event_index)
        return
    phase, round_in_phase, turn_count, player_idx, actions_remaining, has_session = parsed
    raw_player = replay_move.get("player_idx")
    if not isinstance(raw_player, int) or raw_player != player_idx:
        anomaly("replay_player_snapshot_mismatch", event_index)
    if num_players is not None and (player_idx < 0 or player_idx >= num_players):
        anomaly("replay_player_out_of_range", event_index)

    before = replay_move.get("before_state")
    if not isinstance(before, dict):
        anomaly("missing_before_snapshot", event_index)
        return
    expected_phase = before.get("phase")
    expected_era = before.get("era")
    if phase not in {expected_phase, expected_era}:
        anomaly("replay_snapshot_mismatch", event_index)
    for field, expected in (
        ("round_in_phase", round_in_phase),
        ("turn_count", turn_count),
        ("current_player", player_idx),
        ("actions_remaining", actions_remaining),
    ):
        if before.get(field) != expected:
            anomaly("replay_snapshot_mismatch", event_index)
            break
    if has_session != (
        before.get("choice_set") is not None
        or before.get("current_action_selections") is not None
    ):
        anomaly("replay_snapshot_mismatch", event_index)


def _replay_events(game: Any, events: Iterable[dict]) -> list[ConfirmedAction]:
    pending: tuple[str, list[tuple[str, Any]]] | None = None
    unbound: deque[ConfirmedAction] = deque()
    all_confirmed: list[ConfirmedAction] = []
    undo_stack: list[ConfirmedAction] = []

    for event_index, event in enumerate(events):
        kind = event["kind"]
        if kind == "start_action":
            action_type = str(event.get("action_type", ""))
            _root_action_index(action_type)
            pending = (action_type, [])
        elif kind == "apply_choice":
            if pending is None:
                raise ValueError(f"apply_choice at event {event_index} has no action session")
            choice_kind = event.get("choice_kind")
            if not isinstance(choice_kind, str) or not choice_kind:
                raise ValueError(f"apply_choice at event {event_index} has no choice kind")
            pending[1].append((choice_kind, event.get("value")))
        elif kind == "cancel_action":
            pending = None
        elif kind == "confirm_action":
            if pending is None:
                raise ValueError(f"confirm_action at event {event_index} has no action session")
            action_type, choices = pending
            # The first state call establishes the same decision boundary that
            # the browser used; legal enumeration is intentionally second.
            state_record = game.get_training_state()
            legal_record = game.get_legal_actions()
            before_save = game.save_state()
            payload = _choices_to_payload(action_type, choices)
            payload["defer_end_turn"] = True
            try:
                game.step_composite_action(payload)
            except Exception as error:
                raise RuntimeError(
                    f"failed to replay confirmed action at event {event_index} "
                    f"({action_type}, {choices!r}): {error}"
                ) from error
            record = ConfirmedAction(
                action_type=action_type,
                choices=list(choices),
                state_record=state_record,
                legal_record=legal_record,
                before_save=before_save,
                position_key=None,
            )
            all_confirmed.append(record)
            unbound.append(record)
            undo_stack.append(record)
            pending = None
        elif kind == "replay_move":
            if not unbound:
                raise ValueError(f"replay_move at event {event_index} has no preceding confirmation")
            record = unbound.popleft()
            replay_move = event
            record.replay_move = replay_move
            record.position_key = (
                str(replay_move["position_key"])
                if replay_move.get("position_key") is not None
                else None
            )
            if _root_action_index(str(replay_move.get("action_type", ""))) != _root_action_index(
                record.action_type
            ):
                raise ValueError(
                    f"replay_move at event {event_index} action type disagrees with confirmation"
                )
            _validate_replay_binding(record)
        elif kind == "undo_last_action":
            while undo_stack and undo_stack[-1].undone:
                undo_stack.pop()
            if undo_stack:
                record = undo_stack.pop()
                game.restore_state(record.before_save)
                record.undone = True
                pending = None
        elif kind == "end_turn":
            game.end_turn_for_replay()
            # The engine only permits undo within the current turn.
            undo_stack.clear()
        elif kind == "resolve_shortfall":
            chosen = event.get("chosen_tile_order")
            if not isinstance(chosen, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in chosen
            ):
                raise ValueError(f"resolve_shortfall at event {event_index} has invalid tile order")
            game.step_composite_action(
                {"shortfall_tile_order": list(chosen), "defer_end_turn": True}
            )
        elif kind in {"start_turn", "replay_analysis"}:
            # get_training_state() performs the required automatic start when
            # actions_remaining_in_turn is zero. These persisted markers carry
            # no additional engine state.
            continue
        else:
            raise ValueError(f"unsupported persisted event kind {kind!r}")

    if pending is not None:
        raise ValueError("action_log ended with an unconfirmed action session")
    if unbound:
        raise ValueError(f"{len(unbound)} confirmations have no replay_move binding")
    return all_confirmed


def _validate_replay_binding(record: ConfirmedAction) -> None:
    replay_move = record.replay_move
    assert replay_move is not None
    action_key = replay_move.get("action_key")
    legal_actions = record.legal_record.get("actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("confirmed position has no legal actions")
    if action_key is not None:
        if not isinstance(action_key, str) or not action_key:
            raise ValueError("replay_move action_key must be a non-empty string or null")
        if not any(action.get("key") == action_key for action in legal_actions):
            raise ValueError(f"replay_move action_key {action_key!r} was not legal at its position")


def _build_expert_positions(
    records: Iterable[ConfirmedAction],
    *,
    human_player: int,
    game_index: int,
    game_seed: int,
    outcome: dict,
    policy_loss_weight: float,
    value_target_usable: bool = True,
) -> list[dict]:
    positions: list[dict] = []
    final_shared_win = float(outcome["shared_win_values"][human_player])
    final_margin = int(outcome["victory_point_margins"][human_player])
    final_vp = int(outcome["victory_points"][human_player])
    for position_index, record in enumerate(
        record
        for record in records
        if not record.undone
        and record.replay_move is not None
        and int(record.replay_move.get("player_idx", -1)) == human_player
    ):
        selected_index = _selected_action_index(record)
        actions = []
        for index, action in enumerate(record.legal_record["actions"]):
            key = action.get("key")
            features = action.get("feature_indices")
            if not isinstance(key, str) or not key:
                raise ValueError("legal action key is missing")
            if not isinstance(features, list) or not features:
                raise ValueError("legal action feature_indices are missing")
            chosen = index == selected_index
            actions.append(
                {
                    "index": index,
                    "key": key,
                    "feature_indices": [int(value) for value in features],
                    "visits": 1 if chosen else 0,
                    "policy_target": 1.0 if chosen else 0.0,
                    "search_policy_target": 1.0 if chosen else 0.0,
                    "expert_action": chosen,
                }
            )
        state_features = record.state_record.get("features")
        if not isinstance(state_features, list) or not state_features:
            raise ValueError("training state has no features")
        replay_move = record.replay_move
        positions.append(
            {
                "record_type": "position",
                "format_version": SELF_PLAY_FORMAT_VERSION,
                "feature_version": int(record.state_record["feature_version"]),
                "game_index": game_index,
                "position_index": position_index,
                "game_seed": game_seed,
                "actor": human_player,
                "phase": _phase_from_position_key(record.position_key),
                "round_in_phase": _round_from_position_key(record.position_key),
                "actions_remaining_in_turn": _actions_remaining_from_position_key(
                    record.position_key
                ),
                "state_features": [float(value) for value in state_features],
                "legal_actions": actions,
                "selected_action_index": selected_index,
                "selected_action_key": actions[selected_index]["key"],
                "search_method": "human_replay",
                "model_id": None,
                "checkpoint_step": None,
                "value_target": {
                    "shared_win": final_shared_win if value_target_usable else 0.0,
                    "victory_point_margin": final_margin if value_target_usable else 0,
                },
                "actor_victory_points": final_vp if value_target_usable else 0,
                "policy_loss_weight": float(policy_loss_weight),
                "value_loss_weight": 1.0 if value_target_usable else 0.0,
                "value_target_usable": bool(value_target_usable),
                "source_type": "human_replay",
                "source_game_id": game_index,
                "source_event_position_key": record.position_key,
            }
        )
    return positions


def _selected_action_index(record: ConfirmedAction) -> int:
    legal_actions = record.legal_record.get("actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("confirmed position has no legal actions")
    replay_move = record.replay_move or {}
    action_key = replay_move.get("action_key")
    if isinstance(action_key, str) and action_key:
        matches = [
            index
            for index, action in enumerate(legal_actions)
            if action.get("key") == action_key
        ]
        if len(matches) != 1:
            raise ValueError(f"action key {action_key!r} does not identify one legal action")
        return matches[0]

    expected = _normalized_event_choices(record.action_type, record.choices)
    matches = [
        index
        for index, action in enumerate(legal_actions)
        if expected == _normalized_legal_choices(action)
        and _action_root_matches(record.action_type, action)
    ]
    if len(matches) != 1:
        # A few old records omit a choice kind for an automatically selected
        # card. Compare the stable semantic values as a compatibility fallback.
        value_matches = [
            index
            for index, action in enumerate(legal_actions)
            if _choice_values(expected) == _choice_values(_normalized_legal_choices(action))
            and _action_root_matches(record.action_type, action)
        ]
        if len(value_matches) == 1:
            return value_matches[0]
        semantic_matches = [
            index
            for index, action in enumerate(legal_actions)
            if _semantic_choice_signature(record, expected)
            == _semantic_legal_signature(action)
            and _action_root_matches(record.action_type, action)
        ]
        if semantic_matches:
            return semantic_matches[0]
        raise ValueError(
            "could not identify the confirmed human action among legal actions: "
            f"type={record.action_type!r}, choices={record.choices!r}, expected={expected!r}, "
            f"matches={matches}"
        )
    return matches[0]


def _semantic_choice_signature(
    record: ConfirmedAction, values: list[tuple[str, int | None]]
) -> list[tuple[str, int | None]]:
    """Replace card slots with card-type ids for old deduplicated legal lists."""
    hand: list[dict] = []
    replay_move = record.replay_move or {}
    before_state = replay_move.get("before_state")
    player_idx = replay_move.get("player_idx")
    if isinstance(before_state, dict) and isinstance(player_idx, int):
        players = before_state.get("players")
        if isinstance(players, list) and 0 <= player_idx < len(players):
            candidate = players[player_idx]
            if isinstance(candidate, dict) and isinstance(candidate.get("hand"), list):
                hand = [card for card in candidate["hand"] if isinstance(card, dict)]
    result: list[tuple[str, int | None]] = []
    for kind, value in values:
        if kind == "card" and value is not None and 0 <= value < len(hand):
            result.append(("card_type", _card_type_index(hand[value].get("card_type"))))
        else:
            result.append((kind, value))
    return result


def _semantic_legal_signature(action: dict) -> list[tuple[str, int | None]]:
    values = _normalized_legal_choices(action)
    discard_types = action.get("discard_card_types")
    if not isinstance(discard_types, list):
        discard_types = []
    result: list[tuple[str, int | None]] = []
    card_index = 0
    for kind, value in values:
        if kind == "card":
            card_type = discard_types[card_index] if card_index < len(discard_types) else None
            result.append(("card_type", int(card_type) if isinstance(card_type, int) else None))
            card_index += 1
        else:
            result.append((kind, value))
    return result


_TOWN_CARD_TYPES = {
    name.lower(): index
    for index, name in enumerate(
        (
            "Stafford",
            "BurtonUponTrent",
            "Cannock",
            "Tamworth",
            "Walsall",
            "Leek",
            "StokeOnTrent",
            "Stone",
            "Uttoxeter",
            "Belper",
            "Derby",
            "Coalbrookdale",
            "Wolverhampton",
            "Dudley",
            "Kidderminster",
            "Worcester",
            "Birmingham",
            "Nuneaton",
            "Coventry",
            "Redditch",
        )
    )
}


def _card_type_index(raw_card_type: Any) -> int | None:
    if not isinstance(raw_card_type, str):
        return None
    if raw_card_type == "WildLocation":
        return 27
    if raw_card_type == "WildIndustry":
        return 28
    if raw_card_type.startswith("Location(") and raw_card_type.endswith(")"):
        return _TOWN_CARD_TYPES.get(raw_card_type[9:-1].lower())
    if raw_card_type.startswith("Industry(") and raw_card_type.endswith(")"):
        names = [part.strip().lower() for part in raw_card_type[9:-1].split(",") if part.strip()]
        if len(names) == 1 and names[0] in INDUSTRIES:
            return 20 + INDUSTRIES[names[0]]
        if set(names) == {"goods", "cotton"}:
            return 26
        return 26
    return None


def _choices_to_payload(action_type: str, choices: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"root_action": _root_action_index(action_type)}
    industry_count = 0
    road_count = 0
    for kind, raw_value in choices:
        if kind == "card":
            payload.setdefault("card", []).append(_as_int(raw_value, "card"))
        elif kind in {"industry", "second_industry", "free_development"}:
            value = _industry_value(raw_value)
            if kind == "second_industry" or kind == "free_development":
                payload.setdefault("second_industry", []).append(value)
            elif action_type_key(action_type) == "developdouble" and industry_count > 0:
                payload.setdefault("second_industry", []).append(value)
            else:
                payload.setdefault("industry", []).append(value)
            industry_count += 1
        elif kind == "build_location":
            payload.setdefault("build_location", []).append(
                _as_int(raw_value, "build_location")
            )
        elif kind in {"road", "second_road"}:
            value = _as_int(raw_value, "road")
            if kind == "second_road" or (
                action_type_key(action_type) == "buildrailroad" and road_count > 0
            ):
                payload.setdefault("second_road", []).append(value)
            else:
                payload.setdefault("road", []).append(value)
            road_count += 1
        elif kind == "network_mode":
            payload.setdefault("network_mode", []).append(_network_mode_value(raw_value))
        elif kind == "coal_source":
            payload.setdefault("coal_sources", []).append(
                _resource_source_value(raw_value)
            )
        elif kind == "iron_source":
            payload.setdefault("iron_sources", []).append(
                _resource_source_value(raw_value)
            )
        elif kind == "beer_source":
            payload.setdefault("beer_sources", []).append(
                _beer_source_value(raw_value)
            )
        elif kind == "action_beer_source":
            payload.setdefault("action_beer_source", []).append(
                _action_beer_source_value(raw_value)
            )
        elif kind == "sell_target":
            payload.setdefault("sell_targets", []).append(
                _as_int(raw_value, "sell_target")
            )
        else:
            raise ValueError(f"unsupported persisted choice kind {kind!r}")
    return payload


def _normalized_event_choices(
    action_type: str, choices: list[tuple[str, Any]]
) -> list[tuple[str, int | None]]:
    normalized: list[tuple[str, int | None]] = []
    # Preserve the original choice order while using the same encoded values
    # consumed by the Rust composite-action selector.
    for kind, raw_value in choices:
        if kind == "card":
            normalized.append(("card", _as_int(raw_value, "card")))
        elif kind in {"industry", "second_industry", "free_development"}:
            value = _industry_value(raw_value)
            normalized.append(("industry", value))
        elif kind in {"road", "second_road"}:
            normalized.append(("road", _as_int(raw_value, "road")))
        elif kind == "network_mode":
            normalized.append(("network_mode", _network_mode_value(raw_value)))
        elif kind == "build_location":
            normalized.append(("build_location", _as_int(raw_value, "build_location")))
        elif kind == "coal_source":
            normalized.append(("coal_source", _resource_source_value(raw_value)))
        elif kind == "iron_source":
            normalized.append(("iron_source", _resource_source_value(raw_value)))
        elif kind == "beer_source":
            normalized.append(("beer_source", _beer_source_value(raw_value)))
        elif kind == "action_beer_source":
            normalized.append(("action_beer_source", _action_beer_source_value(raw_value)))
        elif kind == "sell_target":
            normalized.append(("sell_target", _as_int(raw_value, "sell_target")))
        else:
            raise ValueError(f"unsupported persisted choice kind {kind!r}")
    return normalized


def _normalized_legal_choices(action: dict) -> list[tuple[str, int | None]]:
    choices = action.get("choices")
    if not isinstance(choices, list):
        raise ValueError("legal action choices are missing")
    normalized: list[tuple[str, int | None]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("legal action choice is not an object")
        kind = choice.get("kind")
        if kind == "confirm":
            continue
        if not isinstance(kind, str):
            raise ValueError("legal action choice kind is missing")
        value = choice.get("value")
        if kind in {"industry", "second_industry", "free_development"}:
            normalized.append(("industry", _as_int(value, kind)))
        elif kind in {"road", "second_road"}:
            normalized.append(("road", _as_int(value, kind)))
        elif kind in {"card", "build_location", "network_mode", "coal_source", "iron_source", "beer_source", "action_beer_source", "sell_target"}:
            normalized.append((kind, _as_int(value, kind)))
        else:
            raise ValueError(f"unsupported legal choice kind {kind!r}")
    return normalized


def _action_root_matches(action_type: str, action: dict) -> bool:
    expected = _root_action_index(action_type)
    raw = action.get("root_action")
    return raw is None or int(raw) == expected


def _choice_values(values: list[tuple[str, int | None]]) -> list[int | None]:
    return [value for _, value in values]


def _root_action_index(action_type: str) -> int:
    key = action_type_key(action_type)
    try:
        return ROOT_ACTIONS[key]
    except KeyError as error:
        raise ValueError(f"unsupported action type {action_type!r}") from error


def action_type_key(action_type: str) -> str:
    return "".join(character for character in action_type.lower() if character.isalnum())


def _industry_value(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("industry value must not be boolean")
    if isinstance(value, int):
        if 0 <= value < 6:
            return value
        raise ValueError(f"industry index {value} is out of range")
    if isinstance(value, str):
        key = value.lower().replace("_", "")
        if key in INDUSTRIES:
            return INDUSTRIES[key]
    raise ValueError(f"invalid industry value {value!r}")


def _network_mode_value(value: Any) -> int:
    if isinstance(value, str):
        key = value.lower()
        if key == "single":
            return 0
        if key == "double":
            return 1
    return _as_int(value, "network_mode")


def _resource_source_value(value: Any) -> int:
    if isinstance(value, str):
        if value.lower() == "market":
            return MARKET_SOURCE
        raise ValueError(f"invalid resource source {value!r}")
    if isinstance(value, dict) and len(value) == 1:
        key, raw_location = next(iter(value.items()))
        if key == "Building":
            location = _as_int(raw_location, "Building source")
            if not 0 <= location < N_BUILD_LOCATIONS:
                raise ValueError("building resource source is out of range")
            return location
    raise ValueError(f"invalid resource source {value!r}")


def _beer_source_value(value: Any) -> int:
    if isinstance(value, dict) and len(value) == 1:
        key, raw_location = next(iter(value.items()))
        if key == "Building":
            location = _as_int(raw_location, "Building beer source")
            if 0 <= location < N_BUILD_LOCATIONS:
                return location
        if key == "TradePost":
            slot = _as_int(raw_location, "TradePost beer source")
            if 0 <= slot < N_TRADE_POST_SLOTS:
                return N_BUILD_LOCATIONS + slot
    raise ValueError(f"invalid beer source {value!r}")


def _action_beer_source_value(value: Any) -> int:
    if isinstance(value, dict) and len(value) == 1:
        key, raw_location = next(iter(value.items()))
        location = _as_int(raw_location, "brewery beer source")
        if not 0 <= location < N_BUILD_LOCATIONS:
            raise ValueError("brewery beer source is out of range")
        if key == "OwnBrewery":
            return location
        if key == "OpponentBrewery":
            return N_BUILD_LOCATIONS + location
    raise ValueError(f"invalid action beer source {value!r}")


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _phase_from_position_key(position_key: str | None) -> str:
    if not position_key:
        return "unknown"
    return position_key.split(":", 1)[0].lower()


def _round_from_position_key(position_key: str | None) -> int:
    if not position_key:
        return 0
    parts = position_key.split(":")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def _actions_remaining_from_position_key(position_key: str | None) -> int:
    if not position_key:
        return 0
    parts = position_key.split(":")
    try:
        return int(parts[4])
    except (IndexError, ValueError):
        return 0


def _installed_crate_version() -> str:
    try:
        return importlib.metadata.version("fast_brass")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _write_json_line(handle: Any, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


if __name__ == "__main__":
    main()
