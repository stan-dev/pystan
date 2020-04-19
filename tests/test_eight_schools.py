import numpy as np
import pandas as pd
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
    "J": 8,
    "y": (28, 8, -3, 7, -1, 1, 18, 12),
    "sigma": (15, 10, 16, 11, 9, 11, 10, 18),
}


@pytest.fixture(scope="module")
def posterior():
    """Build (compile) a simple model."""
    return stan.build(program_code, data=schools_data)


def test_eight_schools_build(posterior):
    """Verify eight schools compiles."""
    assert posterior is not None


def test_eight_schools_build_numpy(posterior):
    """Verify eight schools compiles."""
    schools_data_alt = {
        "J": 8,
        "y": np.array([28, 8, -3, 7, -1, 1, 18, 12]),
        "sigma": pd.Series([15, 10, 16, 11, 9, 11, 10, 18], name="sigma"),
    }
    posterior_alt = stan.build(program_code, data=schools_data_alt)
    assert posterior_alt is not None


def test_eight_schools_sample(posterior):
    """Sample from a simple model."""
    num_chains, num_samples = 2, 200
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=num_samples)
    num_flat_params = schools_data["J"] * 2 + 2
    assert fit.values.shape == (len(fit.sample_and_sampler_param_names) + num_flat_params, num_samples, num_chains,)
    df = fit.to_frame()
    assert "eta.1" in df.columns
    assert len(df["eta.1"]) == num_samples * num_chains
    assert fit["eta"].shape == (schools_data["J"], num_chains * num_samples)


def test_eight_schools_parameter_indexes(posterior):
    num_chains, num_samples = 1, 200
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=num_samples)
    offset = len(fit.sample_and_sampler_param_names)
    assert fit._parameter_indexes("mu") == (offset + 0,)
    assert fit._parameter_indexes("tau") == (offset + 1,)
    assert fit._parameter_indexes("eta") == tuple(offset + i for i in (2, 3, 4, 5, 6, 7, 8, 9))
    assert fit._parameter_indexes("theta") == tuple(offset + i for i in (10, 11, 12, 13, 14, 15, 16, 17))
