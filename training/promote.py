"""Promote a checkpoint only after a passing Rust evaluation gate.

This command is intentionally independent of the browser/UI stack.  It treats
the evaluation report as the authority for the statistical gate, but verifies
the report's model identities against the bytes on disk before changing the
champion pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PromotionRejected(ValueError):
    """Raised when a candidate does not satisfy the immutable promotion gate."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically promote a checkpoint after a passing Rust evaluation report."
    )
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint")
    parser.add_argument("--champion", required=True, help="Current champion checkpoint")
    parser.add_argument("--report", required=True, help="Full JSON evaluation report")
    parser.add_argument(
        "--output",
        help="Destination champion path (defaults to --champion)",
    )
    parser.add_argument(
        "--manifest",
        help="Promotion manifest path (defaults to <output>.promotion.json)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm replacement when the destination already exists",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = promote_candidate(
        args.candidate,
        args.champion,
        args.report,
        output=args.output,
        manifest=args.manifest,
        confirm=args.confirm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def promote_candidate(
    candidate: str | Path,
    champion: str | Path,
    report: str | Path,
    *,
    output: str | Path | None = None,
    manifest: str | Path | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Validate and atomically promote ``candidate``.

    The existing champion is copied to a sibling backup before replacement.
    ``confirm`` is required only when the destination already exists, which
    prevents a mistaken report from silently replacing a live model.
    """
    candidate_path = Path(candidate).resolve()
    champion_path = Path(champion).resolve()
    destination = Path(output).resolve() if output is not None else champion_path
    report_path = Path(report).resolve()
    manifest_path = (
        Path(manifest).resolve()
        if manifest is not None
        else destination.with_name(f"{destination.name}.promotion.json")
    )

    for label, path in (
        ("candidate", candidate_path),
        ("champion", champion_path),
        ("evaluation report", report_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if candidate_path == champion_path:
        raise PromotionRejected("candidate and champion must be different files")
    if destination.exists() and not confirm:
        raise PromotionRejected(
            f"refusing to replace existing destination without --confirm: {destination}"
        )
    if manifest_path.exists():
        raise PromotionRejected(f"promotion manifest already exists: {manifest_path}")

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionRejected(f"could not read evaluation report: {report_path}") from error
    if not isinstance(report_payload, dict):
        raise PromotionRejected("evaluation report must be a JSON object")
    summary = report_payload.get("summary")
    if not isinstance(summary, dict):
        raise PromotionRejected("evaluation report is missing summary")
    if summary.get("promote") is not True:
        raise PromotionRejected(
            "evaluation gate did not pass: " + json.dumps(summary, sort_keys=True)
        )
    games = summary.get("games")
    minimum_games = summary.get("minimum_games")
    lower_bound = summary.get("score_delta_lower_95")
    promotion_margin = summary.get("promotion_margin", 0.0)
    if (
        not isinstance(games, int)
        or isinstance(games, bool)
        or not isinstance(minimum_games, int)
        or isinstance(minimum_games, bool)
        or games < minimum_games
        or not isinstance(lower_bound, (int, float))
        or isinstance(lower_bound, bool)
        or not isinstance(promotion_margin, (int, float))
        or isinstance(promotion_margin, bool)
        or float(lower_bound) <= float(promotion_margin)
    ):
        raise PromotionRejected("passing evaluation summary is internally inconsistent")

    candidate_id = _file_model_id(candidate_path)
    champion_id = _file_model_id(champion_path)
    if report_payload.get("candidate_model_id") != candidate_id:
        raise PromotionRejected(
            "candidate model ID in the report does not match the checkpoint bytes"
        )
    if report_payload.get("champion_model_id") != champion_id:
        raise PromotionRejected(
            "champion model ID in the report does not match the checkpoint bytes"
        )
    _validate_report_source_name(report_payload.get("candidate"), candidate_path, "candidate")
    _validate_report_source_name(report_payload.get("champion"), champion_path, "champion")

    temporary = destination.with_name(f".{destination.name}.promote-{os.getpid()}.tmp")
    if temporary.exists():
        raise PromotionRejected(f"promotion temporary file already exists: {temporary}")

    backup_path: Path | None = None
    if destination.exists():
        if _file_model_id(destination) != champion_id:
            raise PromotionRejected(
                "existing destination is not the champion evaluated by the report"
            )
        # Keep a recoverable copy next to the destination.  Refuse an existing
        # name instead of overwriting a prior backup from another promotion.
        backup_path = destination.with_name(
            f"{destination.stem}.pre-promotion-{champion_id[7:19]}{destination.suffix}"
        )
        if backup_path.exists():
            raise PromotionRejected(f"promotion backup already exists: {backup_path}")
        shutil.copy2(destination, backup_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(candidate_path, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    result: dict[str, Any] = {
        "event": "checkpoint_promoted",
        "candidate": str(candidate_path),
        "candidate_model_id": candidate_id,
        "champion_before": str(champion_path),
        "champion_before_model_id": champion_id,
        "destination": str(destination),
        "backup": str(backup_path) if backup_path is not None else None,
        "report": str(report_path),
        "summary": summary,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, result)
    return result


def _file_model_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_report_source_name(
    raw_value: object, expected: Path, label: str
) -> None:
    if not isinstance(raw_value, str) or not raw_value:
        raise PromotionRejected(f"evaluation report is missing {label} source")
    # Reports produced inside Docker use /work paths while promotion commonly
    # runs on Windows.  The basename is the portable identity we can verify.
    reported_name = raw_value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if reported_name != expected.name:
        raise PromotionRejected(
            f"evaluation report {label} source does not name {expected.name!r}"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PromotionRejected(f"promotion manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise PromotionRejected(f"promotion manifest temporary file already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
