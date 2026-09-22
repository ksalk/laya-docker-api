"""parse_args robustness (#8): lifespan parses env-only, never sys.argv;
bad integer env vars fail with a clear message instead of a raw ValueError."""

import sys

import pytest

from app.main import parse_args


@pytest.fixture(autouse=True)
def clean_laya_env(monkeypatch):
    for name in (
        "LAYA_PRELOAD",
        "LAYA_MAX_LOADED",
        "LAYA_DEVICE",
        "LAYA_HOST",
        "LAYA_PORT",
        "LAYA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_with_empty_argv():
    args = parse_args(argv=[])
    assert args.max_loaded == 2
    assert args.port == 8120
    assert args.device == "cuda"
    assert args.preload == "english,multilingual"


def test_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_LOADED", "4")
    args = parse_args(argv=["--max-loaded", "3"])
    assert args.max_loaded == 3


def test_env_fallback_with_empty_argv(monkeypatch):
    monkeypatch.setenv("LAYA_PRELOAD", "all")
    args = parse_args(argv=[])
    assert args.preload == "all"


def test_sys_argv_is_ignored_when_argv_given(monkeypatch):
    # The uvicorn scenario: uvicorn's own flags are on sys.argv. Passing
    # argv=[] must make argparse blind to them.
    monkeypatch.setattr(sys, "argv", ["prog", "--totally-unknown", "--host", "evil"])
    args = parse_args(argv=[])
    assert args.host == "0.0.0.0"


def test_bad_max_loaded_env_exits_with_name(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_LOADED", "foo")
    with pytest.raises(SystemExit, match="LAYA_MAX_LOADED"):
        parse_args(argv=[])


def test_empty_port_env_exits_clearly(monkeypatch):
    monkeypatch.setenv("LAYA_PORT", "")
    with pytest.raises(SystemExit, match="LAYA_PORT"):
        parse_args(argv=[])


def test_valid_int_env_parses(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_LOADED", "3")
    assert parse_args(argv=[]).max_loaded == 3
