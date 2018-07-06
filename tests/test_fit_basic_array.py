"""Test model with array parameter."""
import numpy as np
import pytest

import pystan

program_code = """
    data {
      int<lower=2> K;
    }
    parameters {
      real beta[K,1,2];
    }
    model {
      for (k in 1:K)
        beta[k,1,1] ~ normal(0,1);
      for (k in 1:K)
        beta[k,1,2] ~ normal(100,1);
    }
"""
K = 4
num_samples = 1000
num_chains = 3


@pytest.fixture(scope="module")
def posterior():
    return pystan.build(program_code, data={"K": K})


@pytest.fixture(scope="module")
def fit(posterior):
    assert posterior is not None
    return posterior.sample(num_samples=num_samples, num_chains=num_chains)


def test_fit_array_draw_contents(fit):
    """
    Make sure shapes are getting unraveled correctly. Mixing up row-major and
    column-major data is a potential issue.
    """
    beta = fit["beta"]
    assert beta.shape == (K, 1, 2, num_samples * num_chains)
    beta_mean = np.mean(beta, axis=-1)
    assert beta_mean.shape == (K, 1, 2)
    assert np.all(beta_mean[:, 0, 0] < 4)
    assert np.all(beta_mean[:, 0, 1] > 99)
