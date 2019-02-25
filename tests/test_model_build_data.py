"""Test handling of `data` dictionary."""
import copy

import numpy as np
import pytest

import stan


program_code = """
    data {
    int<lower=0> N;
    int<lower=0,upper=1> y[N];
    }
    parameters {
    real<lower=0,upper=1> theta;
    }
    model {
    for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
    """

data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, data=data)


def test_data_wrong_dtype(posterior):
    # pull in posterior to cache compilation
    bad_data = copy.deepcopy(data)
    # float is wrong dtype
    bad_data["y"] = np.array(bad_data["y"], dtype=float)
    assert bad_data["y"].dtype == float
    with pytest.raises(RuntimeError, match=r"int variable contained non-int values"):
        stan.build(program_code, data=bad_data)


def test_data_unmodified(posterior):
    # pull in posterior to cache compilation
    data_with_array = copy.deepcopy(data)
    # `build` will convert data into a list, should not change original
    data_with_array["y"] = np.array(data_with_array["y"], dtype=int)
    assert data_with_array["y"].dtype == int
    stan.build(program_code, data=data_with_array)
    # `data_with_array` should be unchanged
    assert not isinstance(data_with_array["y"], list)
    assert data_with_array["y"].dtype == int
