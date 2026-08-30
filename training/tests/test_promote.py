from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.promote import PromotionRejected, _file_model_id, promote_candidate


class PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.candidate = self.root / "candidate.pt"
        self.champion = self.root / "champion.pt"
        self.report = self.root / "gate.json"
        self.candidate.write_bytes(b"candidate checkpoint")
        self.champion.write_bytes(b"champion checkpoint")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_passing_report_atomically_promotes_and_keeps_backup(self) -> None:
        self._write_report(promote=True)

        result = promote_candidate(
            self.candidate,
            self.champion,
            self.report,
            confirm=True,
        )

        self.assertEqual(self.champion.read_bytes(), b"candidate checkpoint")
        backup = Path(result["backup"])
        self.assertEqual(backup.read_bytes(), b"champion checkpoint")
        manifest = self.champion.with_name("champion.pt.promotion.json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate_model_id"], _file_model_id(self.candidate))
        self.assertEqual(payload["summary"]["promote"], True)

    def test_failed_gate_cannot_change_champion(self) -> None:
        self._write_report(promote=False)

        with self.assertRaisesRegex(PromotionRejected, "did not pass"):
            promote_candidate(
                self.candidate,
                self.champion,
                self.report,
                confirm=True,
            )

        self.assertEqual(self.champion.read_bytes(), b"champion checkpoint")

    def test_report_model_identity_must_match_checkpoint_bytes(self) -> None:
        self._write_report(promote=True, candidate_model_id="sha256:wrong")

        with self.assertRaisesRegex(PromotionRejected, "candidate model ID"):
            promote_candidate(
                self.candidate,
                self.champion,
                self.report,
                confirm=True,
            )

        self.assertEqual(self.champion.read_bytes(), b"champion checkpoint")

    def test_existing_destination_requires_explicit_confirmation(self) -> None:
        self._write_report(promote=True)

        with self.assertRaisesRegex(PromotionRejected, "without --confirm"):
            promote_candidate(self.candidate, self.champion, self.report)

    def _write_report(
        self,
        *,
        promote: bool,
        candidate_model_id: str | None = None,
    ) -> None:
        payload = {
            "candidate": f"/work/output/{self.candidate.name}",
            "candidate_model_id": candidate_model_id or _file_model_id(self.candidate),
            "champion": f"/work/output/{self.champion.name}",
            "champion_model_id": _file_model_id(self.champion),
            "summary": {
                "games": 40,
                "minimum_games": 40,
                "promote": promote,
                "score_delta_lower_95": 0.1 if promote else -0.1,
            },
        }
        self.report.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
