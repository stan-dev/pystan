"""Test model with array parameter."""
import numpy as np
import pytest

import stan

unrestricted_program = """
    parameters {
        real y;
    }
    model {
        y ~ normal(0, 1);
    }
"""

restricted_program = """
    parameters {
        real<lower=0> y;
    }
    model {
        y ~ normal(0, 1);
    }
"""

num_samples = 1000
num_chains = 4


@pytest.fixture
def posterior(request):
    return stan.build(request.param, random_seed=1)


@pytest.mark.parametrize("posterior", [unrestricted_program], indirect=True)
def test_log_prob(posterior):
    """Test log probability against sampled model with unrestriction."""
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
    y = fit["y"][0][0]
    lp__ = fit["lp__"][0][0]
    lp = posterior.log_prob(unconstrained_parameters=[y])
    assert np.allclose(lp__, lp)


@pytest.mark.parametrize("posterior", [restricted_program], indirect=True)
def test_log_prob_restricted(posterior):
    """Test log probability against sampled model with restriction."""
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
    y = fit["y"][0][0]
    y = posterior.unconstrain_pars({"y": y})[0]
    lp__ = fit["lp__"][0][0]
    lp = posterior.log_prob(unconstrained_parameters=[y], adjust_transform=False)
    assert np.allclose(lp__, lp + y)
    adjusted_lp = posterior.log_prob(unconstrained_parameters=[y], adjust_transform=True)
    assert np.allclose(lp__, adjusted_lp)
