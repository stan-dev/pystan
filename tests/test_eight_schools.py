import pystan


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
    'J': 8,
    'y': (28, 8, -3, 7, -1, 1, 18, 12),
    'sigma': (15, 10, 16, 11, 9, 11, 10, 18),
}


def test_eight_schools_compile():
    """Compile a simple model."""
    posterior = pystan.compile(program_code, data=schools_data)
    assert posterior is not None


def test_eight_schools_sample():
    """Compile a simple model."""
    posterior = pystan.compile(program_code, data=schools_data)
    fit = posterior.sample(num_chains=1, num_samples=200, num_warmup=200)
    num_flat_params = schools_data['J'] * 2 + 2
    assert fit.values.shape == (1, 200, num_flat_params)
    df = fit.to_frame()
    assert 'eta.1' in df.columns
    assert len(df['eta.1']) == 200
