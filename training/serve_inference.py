from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .data import FeatureSchema
from .inference import CheckpointEvaluator

MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_VALUE_STATES = 8192
MAX_BATCH_POSITIONS = 256
MAX_BATCH_ACTIONS = 65_536


def main() -> None:
    args = build_parser().parse_args()
    evaluator = CheckpointEvaluator(args.checkpoint, args.device)
    server = InferenceServer((args.host, args.port), evaluator)
    print(
        json.dumps(
            {
                "event": "inference_server_started",
                "host": args.host,
                "port": args.port,
                "device": str(evaluator.device),
                "model_id": evaluator.model_id,
                "checkpoint_step": evaluator.checkpoint_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Fast Brass policy/value checkpoint inference over HTTP."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    return parser


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], evaluator: CheckpointEvaluator):
        super().__init__(address, InferenceRequestHandler)
        self.evaluator = evaluator
        self.inference_lock = threading.Lock()


class InferenceRequestHandler(BaseHTTPRequestHandler):
    server: InferenceServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        evaluator = self.server.evaluator
        self._write_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "model_id": evaluator.model_id,
                "checkpoint_step": evaluator.checkpoint_step,
                "device": str(evaluator.device),
                "feature_version": evaluator.schema.version,
                "state_dim": evaluator.schema.state_dim,
                "action_dim": evaluator.schema.action_dim,
            },
        )

    def do_POST(self) -> None:
        if self.path not in ("/evaluate", "/evaluate-values", "/evaluate-batch"):
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        try:
            request = self._read_json()
            raw_schema = request.get("feature_schema")
            if not isinstance(raw_schema, dict):
                raise ValueError("feature_schema must be an object")
            schema = FeatureSchema.from_schema_dict(raw_schema)
            if self.path == "/evaluate-values":
                states = request.get("states")
                if not isinstance(states, list):
                    raise ValueError("states must be an array")
                if not 1 <= len(states) <= MAX_VALUE_STATES:
                    raise ValueError(
                        f"states must contain between 1 and {MAX_VALUE_STATES} rows"
                    )
                with self.server.inference_lock:
                    prediction = self.server.evaluator.predict_values(
                        states, request_schema=schema
                    )
                response = {"ok": True, **prediction.to_dict()}
            elif self.path == "/evaluate-batch":
                positions = request.get("positions")
                if not isinstance(positions, list) or not 1 <= len(
                    positions
                ) <= MAX_BATCH_POSITIONS:
                    raise ValueError(
                        "positions must contain between 1 and "
                        f"{MAX_BATCH_POSITIONS} rows"
                    )
                request_ids: list[int] = []
                state_records: list[dict] = []
                legal_records: list[dict] = []
                seen_request_ids: set[int] = set()
                total_actions = 0
                for batch_index, position in enumerate(positions):
                    if not isinstance(position, dict):
                        raise ValueError(
                            f"positions[{batch_index}] must be an object"
                        )
                    request_id = position.get("request_id")
                    if (
                        not isinstance(request_id, int)
                        or isinstance(request_id, bool)
                        or request_id < 0
                        or request_id in seen_request_ids
                    ):
                        raise ValueError(
                            "batch request ids must be unique non-negative integers"
                        )
                    state = position.get("state")
                    legal_actions = position.get("legal_actions")
                    if not isinstance(state, dict):
                        raise ValueError(
                            f"positions[{batch_index}].state must be an object"
                        )
                    if not isinstance(legal_actions, dict):
                        raise ValueError(
                            f"positions[{batch_index}].legal_actions must be an object"
                        )
                    actions = legal_actions.get("actions")
                    if not isinstance(actions, list):
                        raise ValueError(
                            f"positions[{batch_index}].legal_actions.actions must be an array"
                        )
                    total_actions += len(actions)
                    if total_actions > MAX_BATCH_ACTIONS:
                        raise ValueError(
                            f"batch contains more than {MAX_BATCH_ACTIONS} legal actions"
                        )
                    seen_request_ids.add(request_id)
                    request_ids.append(request_id)
                    state_records.append(state)
                    legal_records.append(legal_actions)
                with self.server.inference_lock:
                    predictions = self.server.evaluator.predict_batch(
                        state_records, legal_records, request_schema=schema
                    )
                response = {
                    "ok": True,
                    "model_id": self.server.evaluator.model_id,
                    "checkpoint_step": self.server.evaluator.checkpoint_step,
                    "evaluations": [
                        {
                            "request_id": request_id,
                            "action_keys": list(prediction.action_keys),
                            "policy_probabilities": list(
                                prediction.policy_probabilities
                            ),
                            "shared_win_rate": prediction.shared_win_rate,
                            "victory_point_margin": prediction.victory_point_margin,
                        }
                        for request_id, prediction in zip(
                            request_ids, predictions, strict=True
                        )
                    ],
                }
            else:
                state = request.get("state")
                legal_actions = request.get("legal_actions")
                if not isinstance(state, dict):
                    raise ValueError("state must be an object")
                if not isinstance(legal_actions, dict):
                    raise ValueError("legal_actions must be an object")
                with self.server.inference_lock:
                    prediction = self.server.evaluator.predict(
                        state, legal_actions, request_schema=schema
                    )
                response = {"ok": True, **prediction.to_dict()}
            self._write_json(HTTPStatus.OK, response)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)}
            )
        except Exception as error:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"inference failed: {error}"},
            )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(
                f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes"
            )
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    main()
