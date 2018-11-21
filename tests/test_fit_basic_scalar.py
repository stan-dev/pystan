"""Test model with a scalar parameter."""
import numpy as np
import pytest

import stan

program_code = "parameters {real y;} model {y ~ normal(10, 0.0001);}"
num_samples = 1000
num_chains = 3


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code)


@pytest.fixture(scope="module")
def fit(posterior):
    assert posterior is not None
    return posterior.sample(num_samples=num_samples, num_chains=num_chains)


def test_fit_scalar_draw_order(fit):
    # not much to test here
    assert fit is not None
    offset = len(fit.sample_and_sampler_param_names)
    assert fit._draws.shape == (offset + 1, num_samples, num_chains)
    assert len(fit._draws[:, 0, 0]) == offset + 1
    assert fit._parameter_indexes("y") == (offset + 0,)


def test_fit_scalar_param(fit):
    y = fit["y"]
    assert y.shape == (1, num_samples * num_chains)
    assert 9 < np.mean(y) < 11
