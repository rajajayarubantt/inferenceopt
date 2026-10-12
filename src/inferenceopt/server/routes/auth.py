# SPDX-License-Identifier: Apache-2.0
"""Exchange the launch token for a session cookie."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from inferenceopt.server.auth import COOKIE_NAME, tokens_match

router = APIRouter(prefix="/auth", tags=["auth"])


class SessionRequest(BaseModel):
    token: str


class SessionResponse(BaseModel):
    authenticated: bool


@router.post("/session")
def create_session(body: SessionRequest, request: Request, response: Response) -> SessionResponse:
    if not tokens_match(body.token, request.app.state.token):
        raise HTTPException(status_code=401, detail="invalid token")
    response.set_cookie(
        COOKIE_NAME,
        request.app.state.token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    return SessionResponse(authenticated=True)


@router.delete("/session")
def delete_session(response: Response) -> SessionResponse:
    response.delete_cookie(COOKIE_NAME, path="/")
    return SessionResponse(authenticated=False)
