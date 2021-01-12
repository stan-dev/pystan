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
    constrained_pars = {
        "x": constrained_params[0],
        "y": constrained_params[1],
        "x_mult": constrained_params[2],
        "z": constrained_params[3],
    }
    unconstrained_params = posterior.unconstrain_pars(constrained_pars)
    assert np.allclose([x, y], unconstrained_params)
