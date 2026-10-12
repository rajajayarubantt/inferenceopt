# SPDX-License-Identifier: Apache-2.0
"""Launch the local web server (``inferenceopt ui``)."""

from __future__ import annotations

import ipaddress
import logging
import secrets
import threading
import webbrowser

import uvicorn
from rich.console import Console

from inferenceopt.config.redaction import RedactingFilter, register_secret
from inferenceopt.server.app import create_app
from inferenceopt.server.auth import LOOPBACK_HOSTS
from inferenceopt.services import AppContext


class RemoteBindError(Exception):
    pass


def _is_loopback(host: str) -> bool:
    if host.lower() in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(
    ctx: AppContext,
    *,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool | None = None,
    allow_remote: bool | None = None,
    console: Console | None = None,
) -> None:
    cfg = ctx.config.server
    host = host or cfg.host
    port = port or cfg.port
    allow_remote = cfg.allow_remote if allow_remote is None else allow_remote
    open_browser = cfg.open_browser if open_browser is None else open_browser
    console = console or Console()

    if not _is_loopback(host) and not allow_remote:
        raise RemoteBindError(
            f"refusing to bind to non-loopback address {host!r}; pass --allow-remote "
            "if you really want the UI reachable from other machines (token auth stays on)"
        )

    token = cfg.token.get_secret_value() if cfg.token else secrets.token_urlsafe(32)
    register_secret(token)

    extra_hosts = set() if _is_loopback(host) or host in {"0.0.0.0", "::"} else {host}  # noqa: S104
    app = create_app(ctx, token=token, extra_allowed_hosts=extra_hosts)

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104
    if ":" in display_host:
        display_host = f"[{display_host}]"
    url = f"http://{display_host}:{port}/?token={token}"

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    # Access logs would otherwise print the token from the launch URL on every load.
    logging.getLogger("uvicorn.access").addFilter(RedactingFilter())
    server = uvicorn.Server(config)

    console.print(f"[bold green]InferenceOpt UI[/] running at [link={url}]{url}[/link]")
    console.print("Keep this URL private: it grants control of this machine's experiments.")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    server.run()
