import numpy as np
import pytest

from spanish_nlp.topic import cluster_embeddings


def test_cluster_embeddings_groups_close_vectors():
    texts = ["fútbol", "baloncesto", "economía", "mercados"]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    clusters = cluster_embeddings(
        texts,
        vectors,
        n_topics=2,
        random_state=42,
    )

    groups = {frozenset(member.text for member in cluster.members) for cluster in clusters}
    assert groups == {
        frozenset({"fútbol", "baloncesto"}),
        frozenset({"economía", "mercados"}),
    }
    assert all(
        cluster.representative in {member.text for member in cluster.members}
        for cluster in clusters
    )


def test_cluster_embeddings_is_deterministic_for_fixed_seed():
    texts = ["a", "b", "c", "d"]
    vectors = np.array([[1, 0], [0.8, 0.2], [0, 1], [0.2, 0.8]], dtype=np.float32)

    first = cluster_embeddings(texts, vectors, n_topics=2, random_state=7)
    second = cluster_embeddings(texts, vectors, n_topics=2, random_state=7)

    assert first == second


def test_n_topics_cannot_exceed_samples():
    with pytest.raises(ValueError, match="cannot exceed"):
        cluster_embeddings(
            ["only"],
            np.array([[1.0, 0.0]], dtype=np.float32),
            n_topics=2,
        )
