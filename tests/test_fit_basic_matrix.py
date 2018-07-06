"""Test model with a matrix parameter."""
import pytest

import pystan

# individual draw should look like:
# [ 0 5 0 0 ]
# [ 0 5 0 0 ]
# [ 0 5 0 0 ]
# in column-major order this is:
# [ 0 0 0 5 5 5 0 0 0 0 0 0 ]

program_code = """
     data {
       int<lower=2> K;
       int<lower=1> D;
     }
     parameters {
       matrix[K,D] beta;
     }
     model {
     for (k in 1:K)
         for (d in 1:D)
           beta[k,d] ~ normal(if_else(d==2, 5, 0), 0.000001);
     }
"""
K, D = 3, 4

num_samples = 1000
num_chains = 3


@pytest.fixture(scope="module")
def posterior():
    return pystan.build(program_code, data={"K": K, "D": D})


@pytest.fixture(scope="module")
def fit(posterior):
    assert posterior is not None
    return posterior.sample(num_samples=num_samples, num_chains=num_chains)


def test_fit_matrix_draw_order(fit):
    assert fit is not None
    assert fit._draws.shape == (K * D, num_samples, num_chains)
    assert len(fit._draws[:, 0, 0]) == K * D
    assert fit._parameter_indexes("beta") == tuple(range(K * D))


def test_fit_matrix_draw_contents(fit):
    assert fit is not None
    assert fit._draws.shape == (K * D, num_samples, num_chains)
    chain = fit._draws[:, :, 0]
    # stored in column-major order
    assert -1 < chain[0, :].mean() < 1
    assert -1 < chain[1, :].mean() < 1
    assert -1 < chain[2, :].mean() < 1
    assert 4 < chain[3, :].mean() < 6
    assert 4 < chain[4, :].mean() < 6
    assert 4 < chain[5, :].mean() < 6
    assert -1 < chain[6, :].mean() < 1
    assert -1 < chain[7, :].mean() < 1
    assert -1 < chain[8, :].mean() < 1
    assert -1 < chain[9, :].mean() < 1
    assert -1 < chain[10, :].mean() < 1
    assert -1 < chain[11, :].mean() < 1
    beta = fit["beta"]
    assert beta.shape == (K, D, num_samples * num_chains)
