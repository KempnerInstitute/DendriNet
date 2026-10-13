import numpy as np
import pytest

from dendritic_modeling.analysis.utils.streaming_covariance import StreamingCovariance


def test_unequal_shifted_batches_match_direct_covariance():
    samples = np.random.default_rng(0).normal(size=(100, 5)) + 1e6
    moments = StreamingCovariance(5)
    for chunk in (samples[:1], samples[1:80], samples[80:]):
        moments.update(chunk)
    np.testing.assert_allclose(
        moments.covariance, np.cov(samples, rowvar=False), atol=1e-9
    )
    assert moments.n == 100


def test_empty_invalid_and_single_observation_batches():
    moments = StreamingCovariance(2)
    moments.update(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="two observations"):
        _ = moments.covariance
    moments.update([[1.0, 2.0]])
    for batch in ([[1.0]], [[np.nan, 1.0]]):
        with pytest.raises(ValueError):
            moments.update(batch)
    assert moments.n == 1
