import numpy as np
import pytest

import stan

np.random.seed(1)

program_code = """
data {
  int<lower=0> N;
  int<lower=0> p;
  matrix[N,p] x;
  vector[N] y;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(x * beta, sigma);
}
"""

n, p = 50, 3  # smaller n  than in tests/test_linear_regression.py
X = np.random.normal(size=(n, p))
X = (X - np.mean(X, axis=0)) / np.std(X, ddof=1, axis=0, keepdims=True)
beta_true = (1, 3, 5)
y = np.dot(X, beta_true) + np.random.normal(size=n)

data = {"N": n, "p": p, "x": X, "y": y}


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, data=data, random_seed=1)


def test_repr_fit(posterior):
    fit = posterior.sample(num_chains=4)
    expected = """<stan.Fit>\nParameters:\n    beta: (3,)\n    sigma: ()\nDraws: 4000"""
    assert repr(fit) == expected
