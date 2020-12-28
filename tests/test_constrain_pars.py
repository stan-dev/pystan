"""Test constrain parameters."""
import random

import numpy as np
import pytest

import stan

program_code = """
parameters {
  real x;
  real<lower=0> y;
}
transformed parameters {
    real x_mult = x * 2;
}
generated quantities {
    real z = x + y;
}
"""
num_samples = 1000
num_chains = 4
x = random.uniform(0, 10)
y = random.uniform(0, 10)


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, random_seed=1)


@pytest.mark.parametrize("x", [x])
@pytest.mark.parametrize("y", [y])
def test_log_prob(posterior, x: float, y: float):
    constrained_params = posterior.constrain_pars([x, y])
    assert np.allclose(x, constrained_params[0])
    assert np.allclose(np.exp(y), constrained_params[1])
    assert np.allclose(x * 2, constrained_params[2])
    assert np.allclose(x + np.exp(y), constrained_params[3])
