import pytest

from spanish_nlp.resources import parse_sentiwordnet
from spanish_nlp.sentiment import lookup_score, score_terms


def lexicon():
    return parse_sentiwordnet(
        [
            "bueno\ta 0.75 0.0",
            "malo\ta 0.0 0.8",
            "mejorar\tv 0.5 0.0",
        ]
    )


def test_score_terms_averages_positive_and_negative_scores():
    score = score_terms(
        lexicon(),
        [
            ("bueno", "bueno", "a"),
            ("malo", "malo", "a"),
        ],
    )

    assert score.positive == pytest.approx(0.375)
    assert score.negative == pytest.approx(0.4)
    assert score.net == pytest.approx(-0.025)


def test_lookup_falls_back_to_lemma():
    assert lookup_score(
        lexicon(),
        text="mejorando",
        lemma="mejorar",
        pos="v",
    ) == (0.5, 0.0)


def test_unknown_terms_are_neutral():
    score = score_terms(lexicon(), [("desconocido", "desconocido", "n")])
    assert score.positive == 0.0
    assert score.negative == 0.0
