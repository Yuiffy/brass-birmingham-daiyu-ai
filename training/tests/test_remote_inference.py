from __future__ import annotations

import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from training.remote_inference import RemoteInferenceEvaluator

SCHEMA = {
    "version": 1,
    "state_dim": 4,
    "action_dim": 8,
    "state_blocks": [{"name": "all", "offset": 0, "size": 4}],
    "action_blocks": [{"name": "all", "offset": 0, "size": 8}],
}


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _health() -> dict:
    return {
        "ok": True,
        "model_id": "sha256:remote-model",
        "checkpoint_step": 23,
        "feature_version": 1,
        "state_dim": 4,
        "action_dim": 8,
    }


def _evaluation(action_keys: list[str] | None = None) -> dict:
    return {
        "ok": True,
        "model_id": "sha256:remote-model",
        "checkpoint_step": 23,
        "evaluations": [
            {
                "request_id": 0,
                "action_keys": action_keys or ["action-a", "action-b"],
                "policy_probabilities": [0.75, 0.25],
                "shared_win_rate": 0.6,
                "victory_point_margin": 3.5,
            }
        ],
    }


class RemoteInferenceTests(unittest.TestCase):
    def test_remote_batch_retries_a_transient_connection_failure(self) -> None:
        responses = iter(
            [
                _FakeResponse(_health()),
                URLError(ConnectionRefusedError("temporary refusal")),
                _FakeResponse(_evaluation()),
            ]
        )

        def fake_urlopen(*_args, **_kwargs):
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return result

        with (
            patch("training.remote_inference.urlopen", side_effect=fake_urlopen),
            patch("training.remote_inference.time.sleep") as sleep,
        ):
            evaluator = RemoteInferenceEvaluator(
                "http://inference.test", SCHEMA, retry_backoff_seconds=0.25
            )
            prediction = evaluator.predict(
                {"feature_version": 1, "features": [1.0, 0.0, 0.5, -0.5]},
                {
                    "feature_version": 1,
                    "actions": [
                        {"index": 0, "key": "action-a", "feature_indices": [0]},
                        {"index": 1, "key": "action-b", "feature_indices": [1]},
                    ],
                },
            )

        self.assertEqual(prediction.model_id, "sha256:remote-model")
        sleep.assert_called_once_with(0.25)

    def test_remote_batch_preserves_schema_identity_and_action_order(self) -> None:
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            payload = _health() if len(requests) == 1 else _evaluation()
            return _FakeResponse(payload)

        with patch("training.remote_inference.urlopen", side_effect=fake_urlopen):
            evaluator = RemoteInferenceEvaluator(
                "http://inference.test:8765/", SCHEMA, timeout_seconds=9.0
            )
            prediction = evaluator.predict(
                {"feature_version": 1, "features": [1.0, 0.0, 0.5, -0.5]},
                {
                    "feature_version": 1,
                    "actions": [
                        {"index": 0, "key": "action-a", "feature_indices": [0]},
                        {"index": 1, "key": "action-b", "feature_indices": [1]},
                    ],
                },
                evaluator.schema,
            )

        self.assertEqual(prediction.model_id, "sha256:remote-model")
        self.assertEqual(prediction.checkpoint_step, 23)
        self.assertEqual(prediction.action_keys, ("action-a", "action-b"))
        self.assertEqual(prediction.policy_probabilities, (0.75, 0.25))
        self.assertEqual([request.full_url for request, _ in requests], [
            "http://inference.test:8765/health",
            "http://inference.test:8765/evaluate-batch",
        ])
        self.assertEqual([timeout for _, timeout in requests], [9.0, 9.0])
        request_payload = json.loads(requests[1][0].data)
        self.assertEqual(request_payload["feature_schema"], SCHEMA)
        self.assertEqual(request_payload["positions"][0]["request_id"], 0)

    def test_remote_batch_rejects_reordered_actions(self) -> None:
        responses = iter(
            [_FakeResponse(_health()), _FakeResponse(_evaluation(["action-b", "action-a"]))]
        )
        with patch("training.remote_inference.urlopen", side_effect=lambda *_args, **_kwargs: next(responses)):
            evaluator = RemoteInferenceEvaluator("http://inference.test", SCHEMA)
            with self.assertRaisesRegex(RuntimeError, "stable action-key order"):
                evaluator.predict(
                    {"feature_version": 1, "features": [1.0, 0.0, 0.5, -0.5]},
                    {
                        "feature_version": 1,
                        "actions": [
                            {"index": 0, "key": "action-a", "feature_indices": [0]},
                            {"index": 1, "key": "action-b", "feature_indices": [1]},
                        ],
                    },
                )


if __name__ == "__main__":
    unittest.main()
