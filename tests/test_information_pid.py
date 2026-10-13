"""Minimum-mutual-information partial information decomposition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dendritic_modeling.utils.information.pid import (
    PIDTerms,
    mmi_pid,
    partial_information_decomposition,
)


def _gaussian_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Exact MI of jointly Gaussian samples from their covariance (nats)."""
    joint = np.cov(np.hstack([x, y.reshape(len(y), -1)]), rowvar=False)
    k = x.shape[1]
    _, logdet_x = np.linalg.slogdet(joint[:k, :k])
    _, logdet_y = np.linalg.slogdet(np.atleast_2d(joint[k:, k:]))
    _, logdet = np.linalg.slogdet(joint)
    return 0.5 * (logdet_x + logdet_y - logdet)


def _discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in MI of discrete samples (nats)."""
    rows = [tuple(row) for row in np.asarray(x).reshape(len(x), -1)]
    labels = [tuple(np.atleast_1d(value)) for value in y]
    n = len(rows)
    joint: dict = {}
    for pair in zip(rows, labels):
        joint[pair] = joint.get(pair, 0) + 1
    px = {key: rows.count(key) / n for key in set(rows)}
    py = {key: labels.count(key) / n for key in set(labels)}
    return sum(
        count / n * math.log(count / n / (px[a] * py[b]))
        for (a, b), count in joint.items()
    )


def test_mmi_takes_the_smaller_information_as_redundancy():
    terms = mmi_pid(0.25, 0.5, 1.0)
    assert terms == PIDTerms(
        redundancy=0.25, unique_1=0.0, unique_2=0.25, synergy=0.5, joint=1.0
    )


def test_a_non_finite_information_makes_every_term_nan():
    terms = mmi_pid(float("nan"), 0.5, 1.0)
    assert all(math.isnan(value) for value in (terms.redundancy, terms.synergy))


def test_xor_information_is_purely_synergistic():
    bits = np.array([[a, b] for a in (0, 1) for b in (0, 1)] * 25)
    target = bits[:, 0] ^ bits[:, 1]
    terms = partial_information_decomposition(
        bits[:, 0], bits[:, 1], target, _discrete_mi
    )
    assert terms.redundancy == pytest.approx(0.0, abs=1e-12)
    assert terms.synergy == pytest.approx(math.log(2))


def test_gaussian_sum_matches_the_closed_form():
    rng = np.random.default_rng(0)
    n, noise = 200_000, 0.5
    s1, s2 = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    target = (s1 + s2 + noise * rng.normal(size=(n, 1)))[:, 0]
    terms = partial_information_decomposition(s1, s2, target, _gaussian_mi)

    single = 0.5 * math.log((2 + noise**2) / (1 + noise**2))
    joint = 0.5 * math.log((2 + noise**2) / noise**2)
    assert terms.redundancy == pytest.approx(single, abs=5e-3)
    assert terms.joint == pytest.approx(joint, abs=5e-3)
    assert terms.synergy == pytest.approx(joint - single, abs=1e-2)
    assert terms.unique_1 + terms.unique_2 == pytest.approx(
        abs(terms.unique_1 - terms.unique_2)
    )
