from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

TOPIC_POS = {"NOUN", "PROPN", "VERB"}


@dataclass(frozen=True, slots=True)
class TopicMember:
    text: str
    distance: float


@dataclass(frozen=True, slots=True)
class TopicCluster:
    topic_id: int
    representative: str
    members: tuple[TopicMember, ...]


def cluster_embeddings(
    texts: list[str],
    vectors: np.ndarray,
    *,
    n_topics: int,
    random_state: int = 42,
) -> list[TopicCluster]:
    if not texts:
        return []
    if len(texts) != len(vectors):
        raise ValueError("texts and vectors must contain the same number of items")
    if n_topics < 1:
        raise ValueError("n_topics must be at least 1")
    if n_topics > len(texts):
        raise ValueError("n_topics cannot exceed the number of vectorized sentences")

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")

    model = KMeans(
        n_clusters=n_topics,
        n_init=10,
        random_state=random_state,
    )
    labels = model.fit_predict(matrix)

    clusters: list[TopicCluster] = []
    for topic_id in range(n_topics):
        indices = np.flatnonzero(labels == topic_id)
        centroid = model.cluster_centers_[topic_id]
        ranked = sorted(
            (
                TopicMember(
                    text=texts[index],
                    distance=float(np.linalg.norm(matrix[index] - centroid)),
                )
                for index in indices
            ),
            key=lambda member: member.distance,
        )
        clusters.append(
            TopicCluster(
                topic_id=topic_id,
                representative=ranked[0].text,
                members=tuple(ranked),
            )
        )
    return clusters


def vectorize_sentences(
    nlp: Any,
    documents: list[str],
    *,
    extra_stopwords: set[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    stopwords = {word.lower() for word in getattr(nlp.Defaults, "stop_words", set())}
    stopwords.update(extra_stopwords or set())

    texts: list[str] = []
    vectors: list[np.ndarray] = []

    for document in nlp.pipe(documents):
        for sentence in document.sents:
            token_vectors = [
                np.asarray(token.vector, dtype=np.float32)
                for token in sentence
                if token.pos_ in TOPIC_POS
                and token.has_vector
                and token.text.lower() not in stopwords
                and token.lemma_.lower() not in stopwords
            ]
            if not token_vectors:
                continue

            vector = np.mean(np.stack(token_vectors), axis=0)
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue

            texts.append(sentence.text.strip())
            vectors.append(vector / norm)

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return texts, np.stack(vectors)


def detect_topics(
    nlp: Any,
    documents: list[str],
    *,
    n_topics: int,
    random_state: int = 42,
    extra_stopwords: set[str] | None = None,
) -> list[TopicCluster]:
    texts, vectors = vectorize_sentences(
        nlp,
        documents,
        extra_stopwords=extra_stopwords,
    )
    return cluster_embeddings(
        texts,
        vectors,
        n_topics=n_topics,
        random_state=random_state,
    )
