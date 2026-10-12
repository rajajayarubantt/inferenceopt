# SPDX-License-Identifier: Apache-2.0
"""FastAPI application factory: the API plus the bundled single-page UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from inferenceopt import __version__
from inferenceopt.server.auth import LocalAuthMiddleware
from inferenceopt.server.routes import api_router
from inferenceopt.services import AppContext
from inferenceopt.ui_assets import UI_DIR

_UI_MISSING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>InferenceOpt</title></head>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem">
<h1>InferenceOpt</h1>
<p>The API is running, but this install has no built web UI.</p>
<p>From a source checkout run <code>cd ui &amp;&amp; pnpm install &amp;&amp; pnpm build</code>,
or use <code>pnpm dev</code> for development. API docs: <a href="/api/docs">/api/docs</a>.</p>
</body></html>"""


def _operation_id(route: APIRoute) -> str:
    # Stable, readable ids for the generated TypeScript client.
    return route.name


def create_app(
    ctx: AppContext,
    *,
    token: str,
    extra_allowed_hosts: set[str] | None = None,
    ui_dir: Path = UI_DIR,
) -> FastAPI:
    app = FastAPI(
        title="InferenceOpt",
        version=__version__,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
        generate_unique_id_function=_operation_id,
    )
    app.state.ctx = ctx
    app.state.token = token

    allowed = set(ctx.config.server.allowed_hosts) | (extra_allowed_hosts or set())
    app.add_middleware(LocalAuthMiddleware, token=token, allowed_hosts=allowed)
    app.include_router(api_router)

    root = ui_dir.resolve()
    index = root / "index.html"
    if (root / "assets").is_dir():
        # Vite emits content-hashed file names here, so browsers may cache them forever.
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    def spa(path: str) -> FileResponse | HTMLResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404)
        if not index.is_file():
            return HTMLResponse(_UI_MISSING)
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        # Client-side routes all resolve to the SPA shell; never cache it so upgrades apply.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
