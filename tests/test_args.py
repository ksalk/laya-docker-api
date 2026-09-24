"""parse_args robustness (#8): lifespan parses env-only, never sys.argv;
bad integer env vars fail with a clear message instead of a raw ValueError."""

import sys

import pytest

from app.main import parse_args


@pytest.fixture(autouse=True)
def clean_laya_env(monkeypatch):
    for name in (
        "LAYA_PRELOAD",
        "LAYA_DEVICE",
        "LAYA_HOST",
        "LAYA_PORT",
        "LAYA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_with_empty_argv():
    args = parse_args(argv=[])
    assert args.port == 8120
    assert args.device == "cuda"
    assert args.preload == "english,multilingual"


def test_env_fallback_with_empty_argv(monkeypatch):
    monkeypatch.setenv("LAYA_PRELOAD", "all")
    args = parse_args(argv=[])
    assert args.preload == "all"


def test_stale_max_loaded_env_is_ignored(monkeypatch):
    # Removed setting: an old LAYA_MAX_LOADED in the environment is simply
    # unused, with no compatibility shim or warning.
    monkeypatch.setenv("LAYA_MAX_LOADED", "4")
    args = parse_args(argv=[])
    assert not hasattr(args, "max_loaded")


def test_sys_argv_is_ignored_when_argv_given(monkeypatch):
    # The uvicorn scenario: uvicorn's own flags are on sys.argv. Passing
    # argv=[] must make argparse blind to them.
    monkeypatch.setattr(sys, "argv", ["prog", "--totally-unknown", "--host", "evil"])
    args = parse_args(argv=[])
    assert args.host == "0.0.0.0"


def test_empty_port_env_exits_clearly(monkeypatch):
    monkeypatch.setenv("LAYA_PORT", "")
    with pytest.raises(SystemExit, match="LAYA_PORT"):
        parse_args(argv=[])


def test_valid_int_env_parses(monkeypatch):
    monkeypatch.setenv("LAYA_PORT", "9000")
    assert parse_args(argv=[]).port == 9000
