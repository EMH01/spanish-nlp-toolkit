from functools import lru_cache
from typing import Any

import spacy


@lru_cache(maxsize=4)
def load_nlp(model_name: str) -> Any:
    nlp = spacy.load(model_name)
    if not any(name in nlp.pipe_names for name in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    return nlp
