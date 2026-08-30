from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .schema import FeatureSchema


@dataclass(frozen=True)
class RemotePolicyValuePrediction:
    model_id: str
    checkpoint_step: int
    action_keys: tuple[str, ...]
    policy_probabilities: tuple[float, ...]
    shared_win_rate: float
    victory_point_margin: float
    actor_victory_points: float


MAX_REMOTE_BATCH_POSITIONS = 256
MAX_REMOTE_BATCH_ACTIONS = 65_536


class RemoteInferenceEvaluator:
    def __init__(
        self,
        base_url: str,
        raw_schema: dict,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        parsed = urlsplit(normalized_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("inference URL must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("inference URL must not contain credentials")
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
            or timeout_seconds > 3600.0
        ):
            raise ValueError("inference timeout must be between 0 and 3600 seconds")
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or not 0 <= max_retries <= 10
        ):
            raise ValueError("inference max retries must be between 0 and 10")
        if (
            not math.isfinite(retry_backoff_seconds)
            or retry_backoff_seconds < 0.0
            or retry_backoff_seconds > 10.0
        ):
            raise ValueError("inference retry backoff must be between 0 and 10 seconds")

        self.base_url = normalized_url
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.schema = FeatureSchema.from_schema_dict(raw_schema)
        self.raw_schema = self.schema.to_schema_dict()
        health = self._request_json("GET", "/health")
        if health.get("ok") is not True:
            raise RuntimeError("remote inference health response is not healthy")
        self.model_id = _nonempty_string(health.get("model_id"), "health.model_id")
        self.checkpoint_step = _nonnegative_int(
            health.get("checkpoint_step"), "health.checkpoint_step"
        )
        expected_health = {
            "feature_version": self.schema.version,
            "state_dim": self.schema.state_dim,
            "action_dim": self.schema.action_dim,
        }
        for field, expected in expected_health.items():
            if health.get(field) != expected:
                raise ValueError(
                    f"remote inference {field} mismatch: expected {expected}, "
                    f"got {health.get(field)!r}"
                )

    def predict(
        self,
        state_record: dict,
        legal_record: dict,
        request_schema: FeatureSchema | None = None,
    ) -> RemotePolicyValuePrediction:
        return self.predict_batch(
            [state_record], [legal_record], request_schema=request_schema
        )[0]

    def predict_batch(
        self,
        state_records: list[dict],
        legal_records: list[dict],
        request_schema: FeatureSchema | None = None,
    ) -> tuple[RemotePolicyValuePrediction, ...]:
        if request_schema is not None:
            self.schema.assert_compatible(request_schema, "remote inference request")
        if not isinstance(state_records, list) or not state_records:
            raise ValueError("remote inference requires at least one state")
        if not isinstance(legal_records, list) or len(legal_records) != len(
            state_records
        ):
            raise ValueError(
                "remote inference states and legal actions must have equal lengths"
            )
        predictions: list[RemotePolicyValuePrediction] = []
        for batch_states, batch_legal in _split_batches(
            state_records, legal_records
        ):
            predictions.extend(
                self._predict_batch_chunk(batch_states, batch_legal)
            )
        return tuple(predictions)

    def _predict_batch_chunk(
        self, state_records: list[dict], legal_records: list[dict]
    ) -> tuple[RemotePolicyValuePrediction, ...]:
        positions = [
            {
                "request_id": request_id,
                "state": state,
                "legal_actions": legal,
            }
            for request_id, (state, legal) in enumerate(
                zip(state_records, legal_records, strict=True)
            )
        ]
        response = self._request_json(
            "POST",
            "/evaluate-batch",
            {"feature_schema": self.raw_schema, "positions": positions},
        )
        if response.get("ok") is not True:
            raise RuntimeError("remote inference batch response is not successful")
        if response.get("model_id") != self.model_id:
            raise RuntimeError("remote inference model ID changed")
        if response.get("checkpoint_step") != self.checkpoint_step:
            raise RuntimeError("remote inference checkpoint step changed")
        evaluations = response.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(positions):
            raise RuntimeError("remote inference returned the wrong evaluation count")

        by_request_id: dict[int, dict] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                raise RuntimeError("remote inference evaluation must be an object")
            request_id = evaluation.get("request_id")
            if (
                not isinstance(request_id, int)
                or isinstance(request_id, bool)
                or request_id < 0
                or request_id >= len(positions)
                or request_id in by_request_id
            ):
                raise RuntimeError("remote inference returned an invalid request ID")
            by_request_id[request_id] = evaluation

        predictions: list[RemotePolicyValuePrediction] = []
        for request_id, legal in enumerate(legal_records):
            evaluation = by_request_id.get(request_id)
            if evaluation is None:
                raise RuntimeError("remote inference omitted a requested position")
            actions = legal.get("actions") if isinstance(legal, dict) else None
            if not isinstance(actions, list) or not actions:
                raise ValueError(
                    f"remote inference legal actions {request_id} must be non-empty"
                )
            expected_keys = tuple(action.get("key") for action in actions)
            if any(not isinstance(key, str) or not key for key in expected_keys):
                raise ValueError(
                    f"remote inference legal actions {request_id} contain invalid keys"
                )
            raw_keys = evaluation.get("action_keys")
            if not isinstance(raw_keys, list) or tuple(raw_keys) != expected_keys:
                raise RuntimeError("remote inference changed stable action-key order")
            raw_probabilities = evaluation.get("policy_probabilities")
            if not isinstance(raw_probabilities, list) or len(raw_probabilities) != len(
                expected_keys
            ):
                raise RuntimeError("remote inference returned the wrong policy length")
            probabilities = tuple(
                _finite_float(value, "policy probability")
                for value in raw_probabilities
            )
            if any(value < 0.0 for value in probabilities) or not math.isclose(
                sum(probabilities), 1.0, abs_tol=1e-5
            ):
                raise RuntimeError("remote inference returned invalid policy mass")
            shared_win_rate = _finite_float(
                evaluation.get("shared_win_rate"), "shared-win rate"
            )
            if not 0.0 <= shared_win_rate <= 1.0:
                raise RuntimeError("remote inference returned invalid shared-win rate")
            victory_point_margin = _finite_float(
                evaluation.get("victory_point_margin"), "victory-point margin"
            )
            actor_victory_points = _finite_float(
                evaluation.get("actor_victory_points", 0.0),
                "actor victory points",
            )
            if actor_victory_points < 0.0:
                raise RuntimeError("remote inference returned invalid actor victory points")
            predictions.append(
                RemotePolicyValuePrediction(
                    model_id=self.model_id,
                    checkpoint_step=self.checkpoint_step,
                    action_keys=expected_keys,
                    policy_probabilities=probabilities,
                    shared_win_rate=shared_win_rate,
                    victory_point_margin=victory_point_margin,
                    actor_victory_points=actor_victory_points,
                )
            )
        return tuple(predictions)

    def _request_json(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    status = int(response.status)
                    response_body = response.read()
                break
            except HTTPError as error:
                detail = error.read(1024).decode("utf-8", errors="replace")
                if error.code not in (408, 429, 500, 502, 503, 504) or not self._retry(
                    attempt
                ):
                    raise RuntimeError(
                        f"remote inference {self.base_url}{path} returned HTTP "
                        f"{error.code}: {detail}"
                    ) from error
            except (OSError, URLError) as error:
                if not self._retry(attempt):
                    raise RuntimeError(
                        f"remote inference request to {self.base_url}{path} failed "
                        f"after {attempt + 1} attempts: {error}"
                    ) from error
        if status != 200:
            raise RuntimeError(
                f"remote inference {self.base_url}{path} returned HTTP {status}"
            )
        try:
            decoded = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("remote inference returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("remote inference response must be an object")
        return decoded

    def _retry(self, failed_attempt: int) -> bool:
        if failed_attempt >= self.max_retries:
            return False
        delay = self.retry_backoff_seconds * (2**failed_attempt)
        if delay > 0.0:
            time.sleep(delay)
        return True


def _split_batches(
    state_records: list[dict], legal_records: list[dict]
) -> list[tuple[list[dict], list[dict]]]:
    batches: list[tuple[list[dict], list[dict]]] = []
    batch_states: list[dict] = []
    batch_legal: list[dict] = []
    batch_actions = 0

    for state, legal in zip(state_records, legal_records, strict=True):
        actions = legal.get("actions") if isinstance(legal, dict) else None
        if not isinstance(actions, list):
            raise ValueError("remote inference legal actions must contain an actions array")
        action_count = len(actions)
        if action_count > MAX_REMOTE_BATCH_ACTIONS:
            raise ValueError(
                "remote inference position contains more than "
                f"{MAX_REMOTE_BATCH_ACTIONS} legal actions"
            )
        if batch_states and (
            len(batch_states) >= MAX_REMOTE_BATCH_POSITIONS
            or batch_actions + action_count > MAX_REMOTE_BATCH_ACTIONS
        ):
            batches.append((batch_states, batch_legal))
            batch_states = []
            batch_legal = []
            batch_actions = 0
        batch_states.append(state)
        batch_legal.append(legal)
        batch_actions += action_count

    if batch_states:
        batches.append((batch_states, batch_legal))
    return batches


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{name} must be finite")
    return converted
