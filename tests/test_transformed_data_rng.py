"""Verify that the RNG in the transformed data block uses the overall seed."""
import numpy as np

import stan

program_code = """
data {
  int<lower=0> N;
}
transformed data {
  vector[N] y;
  for (n in 1:N)
    y[n] = normal_rng(0, 1);
}
parameters {
  real mu;
  real<lower = 0> sigma;
}
model {
  y ~ normal(mu, sigma);
}
generated quantities {
  real mean_y = mean(y);
  real sd_y = sd(y);
}
"""

data = {"N": 100}


def test_generated_quantities_seed():
    fit1 = stan.build(program_code, data=data, random_seed=123).sample(num_samples=10)
    fit2 = stan.build(program_code, data=data, random_seed=123).sample(num_samples=10)
    fit3 = stan.build(program_code, data=data, random_seed=456).sample(num_samples=10)
    assert np.allclose(fit1["mean_y"], fit2["mean_y"])
    assert not np.allclose(fit1["mean_y"], fit3["mean_y"])
