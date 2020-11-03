"""Test a "large" version of the schools model.

Introduced in response to a macOS bug that only
triggered when a larger number of parameters were used.
"""
import pytest

import stan

program_code = """
    data {
      int<lower=0> J; // number of schools
      real y[J]; // estimated treatment effects
      real<lower=0> sigma[J]; // s.e. of effect estimates
    }
    parameters {
      real mu;
      real<lower=0> tau;
      real eta[J];
    }
    transformed parameters {
      real theta[J];
      for (j in 1:J)
        theta[j] = mu + tau * eta[j];
    }
    model {
      target += normal_lpdf(eta | 0, 1);
      target += normal_lpdf(y | theta, sigma);
    }
"""
schools_data = {
    "J": 8 * 20,
    "y": (28, 8, -3, 7, -1, 1, 18, 12) * 20,
    "sigma": (15, 10, 16, 11, 9, 11, 10, 18) * 20,
}


@pytest.fixture(scope="module")
def posterior():
    """Build (compile) a simple model."""
    return stan.build(program_code, data=schools_data)


def test_eight_schools_large_sample(posterior):
    num_chains, num_samples = 2, 200
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=num_samples)
    num_flat_params = schools_data["J"] * 2 + 2
    assert fit.values.shape == (len(fit.sample_and_sampler_param_names) + num_flat_params, num_samples, num_chains,)
    df = fit.to_frame()
    assert "eta.1" in df.columns
    assert len(df["eta.1"]) == num_samples * num_chains
    assert fit["eta"].shape == (schools_data["J"], num_chains * num_samples)
