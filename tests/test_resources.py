import pytest

from spanish_nlp.resources import parse_sentiwordnet


def test_zero_zero_entries_are_ignored():
    result = parse_sentiwordnet(["neutral\ta 0 0"])
    assert result == {}


def test_invalid_lexicon_row_has_line_number():
    with pytest.raises(ValueError, match="line 1"):
        parse_sentiwordnet(["broken row"])
