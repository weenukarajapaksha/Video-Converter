"""Speaker-cluster merging, using the similarities and pitches measured on real clips."""

import numpy as np
import pytest

from dubber.stages.diarize import Cluster, merge_speakers


def embed(sims: np.ndarray) -> np.ndarray:
    """Unit vectors whose pairwise cosines equal ``sims`` (Cholesky of the Gram matrix)."""
    return np.linalg.cholesky(sims)


def clusters(labels, durations, f0s, sims):
    vecs = embed(np.array(sims, dtype=float))
    return [Cluster(lab, d, v / np.linalg.norm(v), f) for lab, d, f, v in
            zip(labels, durations, f0s, vecs)]


# Measured on the real 50.6 s film clip (pyannote 3.1 found 5 clusters).
REAL = clusters(
    ["S0", "S1", "S2", "S3", "S4"], [5.8, 11.2, 6.6, 9.2, 2.7], [140.1, 205.2, 109.3, 117.2, 98.0],
    [[1.00, 0.15, 0.17, 0.44, 0.26],
     [0.15, 1.00, 0.25, 0.19, 0.23],
     [0.17, 0.25, 1.00, 0.23, 0.41],
     [0.44, 0.19, 0.23, 1.00, 0.44],
     [0.26, 0.23, 0.41, 0.44, 1.00]])

# Measured on the synthetic clip: exactly two speakers, one female and one male.
SYNTHETIC = clusters(["A", "B"], [11.9, 15.3], [244.7, 154.6], [[1.0, 0.44], [0.44, 1.0]])


def test_over_split_real_clip_is_merged_to_three_speakers():
    mapping, merges = merge_speakers(REAL)
    final = set(mapping.values())
    assert len(final) == 3
    assert mapping["S1"] == "S1"  # the only female voice stays on her own
    assert mapping["S0"] == mapping["S3"]
    assert mapping["S2"] == mapping["S4"]
    assert all(m["reason"] == "similar voice and pitch" for m in merges)


def test_different_speakers_with_similar_embeddings_are_kept_apart_by_pitch():
    mapping, merges = merge_speakers(SYNTHETIC)  # cosine 0.44 alone would merge them
    assert set(mapping.values()) == {"A", "B"} and merges == []


def test_min_speakers_stops_merging():
    mapping, _ = merge_speakers(REAL, min_speakers=4)
    assert len(set(mapping.values())) == 4


def test_tiny_cluster_is_absorbed_only_by_same_gender_host():
    tiny_male = clusters(["big_m", "big_f", "tiny_m"], [10.0, 10.0, 0.8], [110.0, 210.0, 140.0],
                         [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    mapping, merges = merge_speakers(tiny_male)
    # Most similar overall is the female cluster (0.3), but a male fragment can't be hers.
    assert mapping["tiny_m"] == "big_m"
    assert merges[0]["reason"].startswith("tiny cluster")


@pytest.mark.parametrize("f0", [None])
def test_unknown_pitch_requires_stronger_similarity(f0):
    weak = clusters(["a", "b"], [5.0, 5.0], [f0, 120.0], [[1.0, 0.45], [0.45, 1.0]])
    strong = clusters(["a", "b"], [5.0, 5.0], [f0, 120.0], [[1.0, 0.6], [0.6, 1.0]])
    assert len(set(merge_speakers(weak)[0].values())) == 2
    assert len(set(merge_speakers(strong)[0].values())) == 1
