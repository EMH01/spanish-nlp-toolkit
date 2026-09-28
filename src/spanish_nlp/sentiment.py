from dataclasses import dataclass
from typing import Any

from .resources import Lexicon

POS_MAP = {
    "ADJ": "a",
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
}


@dataclass(frozen=True, slots=True)
class PolarityScore:
    positive: float
    negative: float

    @property
    def net(self) -> float:
        return self.positive - self.negative


@dataclass(frozen=True, slots=True)
class SentencePolarity:
    text: str
    score: PolarityScore


@dataclass(frozen=True, slots=True)
class DocumentPolarity:
    score: PolarityScore
    sentences: tuple[SentencePolarity, ...] = ()


def lookup_score(
    lexicon: Lexicon,
    *,
    text: str,
    lemma: str,
    pos: str,
) -> tuple[float, float]:
    key_candidates = (text.lower(), lemma.lower())
    for key in key_candidates:
        if key in lexicon and pos in lexicon[key]:
            return lexicon[key][pos]
    return 0.0, 0.0


def score_terms(
    lexicon: Lexicon,
    terms: list[tuple[str, str, str]],
) -> PolarityScore:
    if not terms:
        return PolarityScore(0.0, 0.0)

    positive = 0.0
    negative = 0.0
    for text, lemma, pos in terms:
        pos_score, neg_score = lookup_score(
            lexicon,
            text=text,
            lemma=lemma,
            pos=pos,
        )
        positive += pos_score
        negative += neg_score

    denominator = len(terms)
    return PolarityScore(
        positive=positive / denominator,
        negative=negative / denominator,
    )


def _terms_from_span(span: Any, stopwords: set[str]) -> list[tuple[str, str, str]]:
    terms: list[tuple[str, str, str]] = []
    for token in span:
        mapped_pos = POS_MAP.get(token.pos_)
        if mapped_pos is None:
            continue
        if token.text.lower() in stopwords or token.lemma_.lower() in stopwords:
            continue
        terms.append((token.text, token.lemma_, mapped_pos))
    return terms


def analyze_document(
    nlp: Any,
    text: str,
    lexicon: Lexicon,
    *,
    include_sentences: bool = True,
    extra_stopwords: set[str] | None = None,
) -> DocumentPolarity:
    stopwords = {word.lower() for word in getattr(nlp.Defaults, "stop_words", set())}
    stopwords.update(extra_stopwords or set())

    document = nlp(text)
    document_score = score_terms(
        lexicon,
        _terms_from_span(document, stopwords),
    )

    if not include_sentences:
        return DocumentPolarity(score=document_score)

    sentence_scores = tuple(
        SentencePolarity(
            text=sentence.text.strip(),
            score=score_terms(
                lexicon,
                _terms_from_span(sentence, stopwords),
            ),
        )
        for sentence in document.sents
    )
    return DocumentPolarity(
        score=document_score,
        sentences=sentence_scores,
    )
