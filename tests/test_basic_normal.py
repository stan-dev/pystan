import pytest

import stan

program_code = "parameters {real y;} model {y ~ normal(0,1);}"


@pytest.fixture(scope="module")
def normal_posterior():
    return stan.build(program_code)


def test_normal_stepsize(normal_posterior):
    fit = normal_posterior.sample(stepsize=0.001)
    assert fit is not None
