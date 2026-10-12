# SPDX-License-Identifier: Apache-2.0
"""Local-server protection.

The server can start processes on this machine, so every API call is authenticated even
on localhost:

* A random session token is printed in the launch URL. The UI exchanges it once for an
  HttpOnly, SameSite=Strict cookie (``POST /api/v1/auth/session``). Scripts may send
  ``Authorization: Bearer <token>`` instead.
* The Host header must be a known name, which blocks DNS-rebinding attacks.
* State-changing requests carrying an Origin header must be same-origin.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp

COOKIE_NAME = "inferenceopt_session"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
PUBLIC_API_PATHS = frozenset({"/api/v1/health", "/api/v1/auth/session"})
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _hostname(host_header: str) -> str:
    """Strip the port from a Host header value, handling IPv6 literals."""
    return (urlsplit(f"//{host_header}").hostname or "").lower()


def tokens_match(supplied: str | None, expected: str) -> bool:
    return supplied is not None and hmac.compare_digest(supplied.encode(), expected.encode())


class LocalAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, token: str, allowed_hosts: set[str]) -> None:
        super().__init__(app)
        self.token = token
        self.allowed_hosts = {h.lower() for h in allowed_hosts} | LOOPBACK_HOSTS

    def _authenticated(self, request: Request) -> bool:
        if tokens_match(request.cookies.get(COOKIE_NAME), self.token):
            return True
        scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and tokens_match(credentials.strip(), self.token)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host = request.headers.get("host", "")
        if _hostname(host) not in self.allowed_hosts:
            return PlainTextResponse("Invalid Host header", status_code=400)

        origin = request.headers.get("origin")
        if (
            request.method in UNSAFE_METHODS
            and origin is not None
            and urlsplit(origin).netloc.lower() != host.lower()
        ):
            return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)

        path = request.url.path
        if (
            path.startswith("/api/")
            and path not in PUBLIC_API_PATHS
            and not self._authenticated(request)
        ):
            return JSONResponse(
                {"detail": "not authenticated: open the URL printed by `inferenceopt ui`"},
                status_code=401,
            )
        return await call_next(request)
