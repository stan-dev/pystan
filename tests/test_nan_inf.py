"""Test serialization of nan and inf values."""
import math

import stan

program_code = """
    parameters {
      real eta;
    }
    transformed parameters {
      real alpha;
      real beta;
      real gamma;
      alpha = not_a_number();
      beta = positive_infinity();
      gamma = negative_infinity();
    }
    model {
      target += normal_lpdf(eta | 0, 1);
    }
"""


def test_nan_inf():
    posterior = stan.build(program_code)
    fit = posterior.sample()
    assert fit is not None
    assert math.isnan(fit["alpha"].ravel()[0])
    assert math.isinf(fit["beta"].ravel()[0])
    assert math.isinf(fit["gamma"].ravel()[0])
