import numpy as np
import pytest

import stan

program_code = """
    data {
    int<lower=0> N;
    array[N] int<lower=0,upper=1> y;
    }
    parameters {
    real<lower=0,upper=1> theta;
    }
    model {
    for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
    """


def test_unsupported_type():
    data = {"N": np.float32(10), "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    with pytest.raises(TypeError, match="Object of type float32 is not JSON serializable"):
        stan.build(program_code, data=data, random_seed=1)


def test_numpy_integer_types():
    data = {"N": np.int64(10), "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    assert stan.build(program_code, data=data, random_seed=1) is not None
    data = {"N": np.int32(10), "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    assert stan.build(program_code, data=data, random_seed=1) is not None
    data = {"N": np.int16(10), "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}
    assert stan.build(program_code, data=data, random_seed=1) is not None
