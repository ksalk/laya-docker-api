import pytest

from app.main import VALID_CHECKPOINTS, parse_preload


def test_single_valid_name():
    assert parse_preload("english") == ["english"]


def test_multiple_valid_names_preserve_order():
    assert parse_preload("multilingual,typed-decisions") == [
        "multilingual",
        "typed-decisions",
    ]


def test_whitespace_is_stripped():
    assert parse_preload("  english , multilingual  ") == ["english", "multilingual"]


def test_case_insensitive():
    assert parse_preload("English,MULTILINGUAL") == ["english", "multilingual"]


def test_all_expands_to_every_checkpoint():
    assert parse_preload("all") == list(VALID_CHECKPOINTS)
    assert parse_preload("ALL") == list(VALID_CHECKPOINTS)


def test_unknown_name_exits():
    with pytest.raises(SystemExit, match="bogus"):
        parse_preload("english,bogus")


def test_empty_string_exits():
    with pytest.raises(SystemExit, match="No checkpoints"):
        parse_preload("")


def test_whitespace_only_exits():
    with pytest.raises(SystemExit, match="No checkpoints"):
        parse_preload("   ")
