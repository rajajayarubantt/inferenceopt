# SPDX-License-Identifier: Apache-2.0
"""A tiny OpenAI-compatible server for tests (standard library only).

    python fake_openai_server.py --port 8080 [--mode ok|crash|oom|never_ready|slow_start]
        [--tokens 8] [--token-delay 0.005] [--usage/--no-usage]

Mimics llama-server's surface: GET /health (503 while "loading", then 200),
GET /v1/models, POST /v1/chat/completions with SSE streaming.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args: argparse.Namespace
ready_at = 0.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *a: object) -> None:  # keep test logs quiet
        sys.stdout.write("request " + (fmt % a) + "\n")
        sys.stdout.flush()

    def _json(self, code: int, body: dict[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            ok = args.mode != "never_ready" and time.monotonic() >= ready_at
            self._json(200 if ok else 503, {"status": "ok" if ok else "loading model"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": args.model_name}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        n = min(int(body.get("max_tokens", 16)), args.tokens)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(args.prefill_delay)
        for i in range(n):
            chunk = {"choices": [{"index": 0, "delta": {"content": f"tok{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(args.token_delay)
        final: dict[str, object] = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        }
        if args.usage:
            final["usage"] = {"prompt_tokens": 12, "completion_tokens": n, "total_tokens": 12 + n}
        final["timings"] = {"predicted_n": n}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


def main() -> None:
    global args, ready_at
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--mode", default="ok", choices=["ok", "crash", "oom", "never_ready", "slow_start"]
    )
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--token-delay", type=float, default=0.005)
    p.add_argument("--prefill-delay", type=float, default=0.02)
    p.add_argument("--model-name", default="fake-model")
    p.add_argument("--usage", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    print("fake server: loading model", flush=True)
    if args.mode == "crash":
        print("error loading model: corrupted file", flush=True)
        sys.exit(1)
    if args.mode == "oom":
        print("ggml_cuda: CUDA error: out of memory", flush=True)
        sys.exit(1)
    ready_at = time.monotonic() + (1.0 if args.mode == "slow_start" else 0.0)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
