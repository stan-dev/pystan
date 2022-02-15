"""Test model with array parameter."""
import random

import numpy as np
import pytest

import stan

program = """
    parameters {
        real y;
    }
    model {
        y ~ normal(0, 1);
    }
"""

num_samples = 1000
num_chains = 4


def gaussian_gradient(x: float, mean: float, var: float) -> float:
    """Analytically evaluate Gaussian gradient."""
    gradient = (mean - x) / (var**2)
    return gradient


@pytest.fixture
def posterior(request):
    return stan.build(request.param, random_seed=1)


@pytest.mark.parametrize("posterior", [program], indirect=True)
def test_grad_log_prob(posterior):
    """Test log probability against sampled model with no restriction."""
    y = random.uniform(0, 10)
    lp__ = gaussian_gradient(y, 0, 1)
    lp = posterior.grad_log_prob(unconstrained_parameters=[y])
    assert np.allclose(lp__, lp)
