from collections.abc import Iterable
from pathlib import Path

Lexicon = dict[str, dict[str, tuple[float, float]]]


def load_stopwords(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    with Path(path).open(encoding="utf-8") as handle:
        return {line.strip().lower() for line in handle if line.strip()}


def parse_sentiwordnet(lines: Iterable[str]) -> Lexicon:
    lexicon: Lexicon = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            word, values = line.split("\t", maxsplit=1)
            pos, positive, negative = values.split()
            positive_score = float(positive)
            negative_score = float(negative)
        except ValueError as exc:
            raise ValueError(f"Invalid lexicon row at line {line_number}: {line}") from exc

        if positive_score == 0.0 and negative_score == 0.0:
            continue

        lexicon.setdefault(word.lower(), {})[pos] = (
            positive_score,
            negative_score,
        )
    return lexicon


def load_sentiwordnet(path: str | Path) -> Lexicon:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_sentiwordnet(handle)
