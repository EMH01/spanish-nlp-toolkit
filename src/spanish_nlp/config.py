import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    spacy_model: str = os.getenv("SPACY_MODEL", "es_core_news_md")
    swn_path: Path | None = Path(value) if (value := os.getenv("SWN_PATH")) else None
    stopwords_path: Path | None = (
        Path(value) if (value := os.getenv("STOPWORDS_PATH")) else None
    )
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
