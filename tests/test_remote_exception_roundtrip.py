from __future__ import annotations

from leantree.utils import RemoteException, deserialize_exception, serialize_exception


class _PicklableCustomError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def test_deserialize_pickled_exception_sets_cause_and_traceback_str() -> None:
    original = ValueError("bad")
    payload = serialize_exception(original)

    err = deserialize_exception(payload, "wrapper")
    assert isinstance(err, RemoteException)
    assert isinstance(err, RuntimeError)
    assert isinstance(err.__cause__, ValueError)
    assert str(err.__cause__) == "bad"
    assert err.traceback_str
    assert "Server traceback" in str(err)


def test_round_trip_custom_exception_falls_back_to_proxy_cause_when_unpickling_fails() -> None:
    # Most custom exception state (custom attrs, non-default __init__) is not
    # reliably preserved by pickling. We still want a useful typed-ish cause.
    original = _PicklableCustomError("custom", code=42)
    payload = serialize_exception(original)

    err = deserialize_exception(payload)
    assert isinstance(err, RemoteException)
    assert err.__cause__ is not None
    assert type(err.__cause__).__name__ == "_PicklableCustomError"
    assert str(err.__cause__) == "custom"


def test_serialize_fallback_includes_exception_info_and_deserialize_sets_proxy_cause() -> None:
    class _NonPicklable(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self._fn = lambda x: x

    payload = serialize_exception(_NonPicklable("nope"))
    assert "exception_info" in payload
    assert payload["exception_info"]["type"] == "_NonPicklable"

    err = deserialize_exception(payload, "wrapper")
    assert isinstance(err, RemoteException)
    assert err.__cause__ is not None
    assert type(err.__cause__).__name__ == "_NonPicklable"
    assert str(err.__cause__) == "nope"
    assert err.traceback_str


def test_deserialize_exception_info_only_uses_builtin_exception_type_when_available() -> None:
    payload = {
        "exception_info": {
            "type": "ValueError",
            "message": "msg",
            "traceback": ["Traceback (most recent call last):\n", "ValueError: msg\n"],
        }
    }
    err = deserialize_exception(payload, "wrapper")
    assert isinstance(err, RemoteException)
    assert isinstance(err.__cause__, ValueError)
    assert str(err.__cause__) == "msg"
    assert err.traceback_str


def test_deserialize_empty_payload_has_no_cause() -> None:
    err = deserialize_exception({}, "wrapper")
    assert isinstance(err, RemoteException)
    assert err.__cause__ is None
