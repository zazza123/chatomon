import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from chatomon.studio._security import Guard, allowed_hosts, cookie_name, is_bind_all

TOKEN = "s3cret-token"
BASE = "http://127.0.0.1:8016"
ORIGIN = {"origin": BASE}


def app() -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    return Starlette(
        routes=[Route("/", ok), Route("/api/x", ok, methods=["GET", "POST", "PATCH"])],
        middleware=[Middleware(Guard, token=TOKEN, host="127.0.0.1", port=8016)],
    )


@pytest.fixture
def client():
    with TestClient(app(), base_url=BASE) as client:
        yield client


def login(client):
    client.get(f"/?token={TOKEN}")


def test_a_foreign_host_is_refused(client):
    assert client.get("/", headers={"host": "evil.example:8016"}).status_code == 400


def test_the_page_without_a_cookie_says_where_the_link_is(client):
    response = client.get("/")
    assert response.status_code == 401
    assert "printed in the terminal" in response.text
    assert "http://" not in response.text
    assert "https://" not in response.text


def test_a_wrong_token_is_refused(client):
    assert client.get("/?token=nope").status_code == 401


def test_the_token_sets_a_strict_cookie_and_leaves_the_address(client):
    response = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{cookie_name(8016)}={TOKEN}")
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert client.get("/").text == "ok"


def test_the_api_needs_the_cookie(client):
    response = client.get("/api/x")
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "Unauthorized"


def test_changes_need_the_studio_origin(client):
    login(client)
    assert client.post("/api/x").status_code == 403
    assert client.post("/api/x", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.patch("/api/x", headers=ORIGIN).text == "ok"
    assert client.get("/api/x").text == "ok"


def test_studios_on_different_ports_use_different_cookies():
    assert cookie_name(8016) != cookie_name(8017)


def test_allowed_hosts():
    assert allowed_hosts("127.0.0.1", 8016) == {"127.0.0.1:8016", "localhost:8016", "[::1]:8016"}
    assert "192.168.1.10:9000" in allowed_hosts("192.168.1.10", 9000)
    assert "[fd00::1]:9000" in allowed_hosts("fd00::1", 9000)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "", "0", "0.0", "::0", "0:0:0:0:0:0:0:0", "::ffff:0.0.0.0"],
)
def test_every_spelling_of_a_bind_all_address_is_recognised(host):
    assert is_bind_all(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "192.168.1.10"])
def test_single_addresses_are_not_bind_all(host):
    assert not is_bind_all(host)
