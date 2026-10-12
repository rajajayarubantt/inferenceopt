# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferenceopt.server import create_app
from inferenceopt.server.auth import COOKIE_NAME
from inferenceopt.services import AppContext
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.server

TOKEN = "test-token-0123456789"


@pytest.fixture
def ui_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ui"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (d / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (d / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return d


@pytest.fixture
def client(ctx: AppContext, ui_dir: Path) -> Iterator[TestClient]:
    app = create_app(ctx, token=TOKEN, ui_dir=ui_dir)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_api_requires_auth(client: TestClient) -> None:
    assert client.get("/api/v1/version").status_code == 401
    assert (
        client.get("/api/v1/version", headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert client.get("/api/v1/version", headers=auth()).status_code == 200


def test_openapi_requires_auth(client: TestClient) -> None:
    assert client.get("/api/openapi.json").status_code == 401


def test_session_cookie_flow(client: TestClient) -> None:
    assert client.post("/api/v1/auth/session", json={"token": "nope"}).status_code == 401
    resp = client.post("/api/v1/auth/session", json={"token": TOKEN})
    assert resp.status_code == 200
    cookie_header = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie_header and "samesite=strict" in cookie_header
    assert client.cookies.get(COOKIE_NAME) == TOKEN
    assert client.get("/api/v1/doctor").status_code == 200
    client.delete("/api/v1/auth/session")
    client.cookies.clear()
    assert client.get("/api/v1/doctor").status_code == 401


def test_dns_rebinding_host_rejected(client: TestClient) -> None:
    resp = client.get("/api/v1/health", headers={"Host": "evil.example:8765"})
    assert resp.status_code == 400


def test_localhost_and_ipv6_hosts_allowed(client: TestClient) -> None:
    for host in ("localhost:8765", "[::1]:8765", "127.0.0.1"):
        assert client.get("/api/v1/health", headers={"Host": host}).status_code == 200


def test_cross_origin_post_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs/validate",
        json={"yaml": "job: {}"},
        headers={**auth(), "Origin": "http://evil.example"},
    )
    assert resp.status_code == 403


def test_same_origin_post_allowed(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs/validate",
        json={"yaml": (REPO_ROOT / "configs/example-job.yaml").read_text(encoding="utf-8")},
        headers={**auth(), "Origin": "http://127.0.0.1:8765"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_config_never_returns_secret_values(monkeypatch: pytest.MonkeyPatch, ui_dir: Path) -> None:
    monkeypatch.setenv("INFERENCEOPT_LLM__API_KEY", "sk-verysecretvalue000000")
    ctx = AppContext.open()
    try:
        app = create_app(ctx, token=TOKEN, ui_dir=ui_dir)
        with TestClient(app, base_url="http://127.0.0.1:8765") as c:
            body = c.get("/api/v1/config", headers=auth()).text
        assert "verysecretvalue" not in body
        assert TOKEN not in body
    finally:
        ctx.close()


def test_spa_routes_and_assets(client: TestClient) -> None:
    assert "id=root" in client.get("/").text
    assert "id=root" in client.get("/jobs/123").text  # client-side route
    assert client.get("/assets/app-abc123.js").text == "console.log(1)"
    assert client.get("/favicon.svg").text == "<svg/>"
    assert client.get("/api/v1/does-not-exist", headers=auth()).status_code == 404


def test_spa_blocks_path_traversal(client: TestClient, ui_dir: Path) -> None:
    (ui_dir.parent / "secret.txt").write_text("top secret", encoding="utf-8")
    resp = client.get("/..%2Fsecret.txt")
    assert "top secret" not in resp.text


def test_missing_ui_shows_help(ctx: AppContext, tmp_path: Path) -> None:
    app = create_app(ctx, token=TOKEN, ui_dir=tmp_path / "empty")
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        assert "no built web UI" in c.get("/").text
