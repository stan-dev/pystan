import pickle
import tempfile

import numpy as np
import pytest

import stan

program_code = "parameters {real y;} model {y ~ normal(0,1);}"


@pytest.fixture(scope="module")
def normal_posterior():
    return stan.build(program_code)


def test_pickle(normal_posterior):
    fit1 = normal_posterior.sample(stepsize=0.001)
    assert fit1["y"] is not None
    with tempfile.TemporaryFile() as fp:
        pickle.dump(fit1, fp)
        fp.seek(0)
        fit2 = pickle.load(fp)
    assert fit2["y"] is not None
    np.testing.assert_array_equal(fit1["y"], fit2["y"])
