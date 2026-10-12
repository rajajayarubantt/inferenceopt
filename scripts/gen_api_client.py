# SPDX-License-Identifier: Apache-2.0
"""Export the OpenAPI schema and regenerate the UI's typed API client.

Run after changing any API route or response model:

    uv run python scripts/gen_api_client.py

CI runs this and fails if ``ui/openapi.json`` or ``ui/src/api/schema.d.ts`` change.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "ui"


def export_openapi() -> Path:
    with tempfile.TemporaryDirectory() as home:
        os.environ["INFERENCEOPT_HOME"] = home
        from inferenceopt.server import create_app
        from inferenceopt.services import AppContext

        ctx = AppContext.open()
        try:
            schema = create_app(ctx, token="unused").openapi()  # noqa: S106
        finally:
            ctx.close()
    # The version changes with every commit; keep the file stable for drift checks.
    schema["info"]["version"] = "0"
    out = UI / "openapi.json"
    out.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def generate_client() -> None:
    pnpm = shutil.which("pnpm")
    cmd = [pnpm] if pnpm else [shutil.which("corepack") or "corepack", "pnpm"]
    subprocess.run([*cmd, "run", "gen:api"], cwd=UI, check=True)  # noqa: S603


def main() -> int:
    path = export_openapi()
    print(f"wrote {path.relative_to(REPO)}")
    if "--schema-only" not in sys.argv:
        generate_client()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
