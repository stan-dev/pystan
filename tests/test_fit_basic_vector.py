"""Test model with a vector parameter."""
import numpy as np
import pytest

import stan

# draws should look like (0, 5, 0)
program_code = """
parameters {
  real beta[3];
}
model {
  beta[1] ~ normal(0, 0.0001);
  beta[2] ~ normal(5, 0.0001);
  beta[3] ~ normal(0, 0.0001);
}
"""

num_samples = 10
num_chains = 2


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, random_seed=123)


@pytest.fixture(scope="module")
def fit(posterior):
    assert posterior is not None
    return posterior.sample(num_samples=num_samples, num_chains=num_chains)


def test_fit_vector_draw_order(fit):
    assert fit is not None
    assert fit._draws.shape == (3, num_samples, num_chains)
    assert len(fit._draws[:, 0, 0]) == 3
    assert fit._parameter_indexes("beta") == tuple(range(3))


def test_fit_vector_draw_contents(fit):
    assert fit is not None
    assert fit._draws.shape == (3, num_samples, num_chains)
    chain = fit._draws[:, :, 0]
    assert -1 < chain[0, :].mean() < 1
    assert 4 < chain[1, :].mean() < 6
    assert -1 < chain[2, :].mean() < 1
    beta = fit["beta"]
    assert beta.shape == (3, num_samples * num_chains)
    beta_mean = np.mean(beta, axis=1)
    assert -1 < beta_mean[0] < 1
    assert 4 < beta_mean[1] < 6
    assert -1 < beta_mean[2] < 1
