import hashlib

import pytest
from pydantic import BaseModel

from chatomon import ChatomonConfigError
from chatomon.observability import ContentPolicy, get_tracer, sha256_text, stable_json


class Point(BaseModel):
    y: int
    x: int


def test_stable_json_sorts_keys_and_handles_models():
    assert stable_json({"b": 1, "a": [Point(y=2, x=1)]}) == '{"a":[{"x":1,"y":2}],"b":1}'


def test_stable_json_falls_back_to_repr():
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert stable_json({"o": Opaque()}) == '{"o":"<opaque>"}'


def test_sha256_text_is_prefixed():
    expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert sha256_text("hello") == expected


def test_none_mode_records_nothing():
    policy = ContentPolicy()
    assert policy.mode == "none"
    assert policy.text("secret") is None
    assert policy.data({"k": "secret"}) is None


def test_hash_mode_records_digests():
    policy = ContentPolicy("hash")
    assert policy.text("secret") == sha256_text("secret")
    assert policy.data({"k": "v"}) == sha256_text('{"k":"v"}')


def test_full_mode_applies_the_redactor():
    policy = ContentPolicy("full", redactor=lambda text: text.replace("Rome", "[CITY]"))
    assert policy.text("It is in Rome.") == "It is in [CITY]."
    assert policy.data({"city": "Rome"}) == '{"city":"[CITY]"}'
    assert policy.text(None) is None


def test_invalid_mode_is_a_config_error():
    with pytest.raises(ChatomonConfigError, match="capture_content"):
        ContentPolicy("everything")  # type: ignore[arg-type]


def test_get_tracer_returns_a_tracer_without_an_sdk():
    tracer = get_tracer()
    with tracer.start_as_current_span("noop") as span:
        assert span is not None
