from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class FilteredShardSummary:
    source: str
    output: str
    minimum_actor_vp: int
    games_written: int
    eligible_player_trajectories: int
    source_positions: int
    positions_written: int


def main() -> None:
    args = build_parser().parse_args()
    summaries = filter_self_play_shards(
        tuple(Path(path) for path in args.shards),
        Path(args.output_prefix),
        args.minimum_actor_vp,
    )
    print(
        json.dumps(
            {
                "minimum_actor_vp": args.minimum_actor_vp,
                "source_positions": sum(item.source_positions for item in summaries),
                "positions_written": sum(item.positions_written for item in summaries),
                "eligible_player_trajectories": sum(
                    item.eligible_player_trajectories for item in summaries
                ),
                "shards": [asdict(item) for item in summaries],
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive policy-training shards containing positions from players "
            "whose final score meets a declared quality threshold."
        )
    )
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--minimum-actor-vp", type=int, required=True)
    return parser


def filter_self_play_shards(
    shard_paths: tuple[Path, ...],
    output_prefix: Path,
    minimum_actor_vp: int,
) -> tuple[FilteredShardSummary, ...]:
    if not shard_paths:
        raise ValueError("at least one source shard is required")
    if minimum_actor_vp < 0:
        raise ValueError("minimum_actor_vp must be non-negative")
    if output_prefix.name in {"", ".", ".."} or output_prefix.suffix == ".jsonl":
        raise ValueError("output_prefix must be a file-name prefix without .jsonl")

    width = max(3, len(str(len(shard_paths) - 1)))
    outputs = tuple(
        Path(f"{output_prefix}-{index:0{width}d}.jsonl").resolve()
        for index in range(len(shard_paths))
    )
    conflicts = [
        candidate
        for output in outputs
        for candidate in (output, Path(f"{output}.partial"))
        if candidate.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "filtered self-play output already exists: "
            + ", ".join(str(path) for path in conflicts)
        )

    return tuple(
        filter_self_play_shard(source, output, minimum_actor_vp)
        for source, output in zip(shard_paths, outputs, strict=True)
    )


def filter_self_play_shard(
    source: Path,
    output: Path,
    minimum_actor_vp: int,
) -> FilteredShardSummary:
    source = source.resolve()
    output = output.resolve()
    partial = Path(f"{output}.partial")
    if not source.is_file():
        raise FileNotFoundError(f"self-play source does not exist: {source}")
    if output.exists() or partial.exists():
        raise FileExistsError(f"filtered self-play output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    games_written = 0
    eligible_trajectories = 0
    source_positions = 0
    positions_written = 0
    current_game: dict | None = None
    current_positions: list[bytes] = []
    current_eligible_players: tuple[int, ...] = ()

    def flush_game(handle: BinaryIO) -> None:
        nonlocal games_written, positions_written
        if current_game is None:
            return
        game = dict(current_game)
        original_positions = game.get("positions")
        if isinstance(original_positions, int) and not isinstance(
            original_positions, bool
        ):
            game["source_positions"] = original_positions
        game["positions"] = len(current_positions)
        game["quality_eligible_players"] = list(current_eligible_players)
        _write_json_line(handle, game)
        for line in current_positions:
            handle.write(line if line.endswith(b"\n") else line + b"\n")
        games_written += 1
        positions_written += len(current_positions)

    try:
        with source.open("rb") as input_handle, partial.open("xb") as output_handle:
            found_metadata = False
            line_number = 0
            for raw_line in input_handle:
                line_number += 1
                if not raw_line.strip():
                    continue
                record = _decode_record(raw_line, source, line_number)
                record_type = record.get("record_type")
                if not found_metadata:
                    if record_type != "metadata":
                        raise ValueError(
                            f"{source}:{line_number}: first record must be metadata"
                        )
                    metadata = dict(record)
                    metadata["derived_dataset"] = "actor_final_vp_quality_filter"
                    metadata["derived_from"] = str(source)
                    metadata["minimum_actor_vp"] = minimum_actor_vp
                    _write_json_line(output_handle, metadata)
                    found_metadata = True
                    continue

                if record_type == "game":
                    flush_game(output_handle)
                    current_game = record
                    current_positions = []
                    victory_points = record.get("victory_points")
                    if not isinstance(victory_points, list) or not victory_points:
                        raise ValueError(
                            f"{source}:{line_number}: game.victory_points is required"
                        )
                    if any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in victory_points
                    ):
                        raise ValueError(
                            f"{source}:{line_number}: victory points must be integers"
                        )
                    current_eligible_players = tuple(
                        index
                        for index, score in enumerate(victory_points)
                        if score >= minimum_actor_vp
                    )
                    eligible_trajectories += len(current_eligible_players)
                elif record_type == "position":
                    if current_game is None:
                        raise ValueError(
                            f"{source}:{line_number}: position precedes its game"
                        )
                    if record.get("game_index") != current_game.get("game_index"):
                        raise ValueError(
                            f"{source}:{line_number}: position game index changed"
                        )
                    actor = record.get("actor")
                    if not isinstance(actor, int) or isinstance(actor, bool):
                        raise ValueError(
                            f"{source}:{line_number}: position.actor is required"
                        )
                    source_positions += 1
                    if actor in current_eligible_players:
                        current_positions.append(raw_line)
                elif record_type == "metadata":
                    raise ValueError(f"{source}:{line_number}: duplicate metadata")
                else:
                    raise ValueError(
                        f"{source}:{line_number}: unknown record type {record_type!r}"
                    )

            if not found_metadata:
                raise ValueError(f"{source}: source shard has no metadata")
            flush_game(output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return FilteredShardSummary(
        source=str(source),
        output=str(output),
        minimum_actor_vp=minimum_actor_vp,
        games_written=games_written,
        eligible_player_trajectories=eligible_trajectories,
        source_positions=source_positions,
        positions_written=positions_written,
    )


def _decode_record(line: bytes, path: Path, line_number: int) -> dict:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return record


def _write_json_line(handle: BinaryIO, record: dict) -> None:
    handle.write(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    handle.write(b"\n")


if __name__ == "__main__":
    main()
