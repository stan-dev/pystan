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


@pytest.fixture(scope="module")
def fit(posterior):
    return posterior.sample(num_chains=4)


def test_bernoulli_sampling_thin(posterior):
    fit = posterior.sample(num_thin=2)
    assert fit["theta"].shape[-1] == 500 * 4


def test_bernoulli_fixed_param(posterior):
    fit = posterior.fixed_param(num_thin=2)
    assert fit["theta"].shape[-1] == 500 * 4
    assert (fit["theta"][0] == fit["theta"]).all()


def test_bernoulli_sampling_invalid_argument(posterior):
    with pytest.raises(TypeError, match=r"'float' object cannot be interpreted as an integer"):
        posterior.sample(num_thin=2.0)


def test_bernoulli_sampling(fit):
    assert fit.num_samples == 1000
    assert fit.param_names == ("theta",)
    assert fit.num_chains == 4

    assert fit._draws.ndim == 3
    assert fit._draws.shape[1] == 1000
    assert fit._draws.shape[2] == 4

    assert len(fit) == 1  # one parameter (theta)

    # for a fit with only one scalar parameter, it is the last one
    assert 0.1 < fit._draws[-1, :, 0].mean() < 0.4
    assert 0.1 < fit._draws[-1, :, 1].mean() < 0.4
    assert 0.1 < fit._draws[-1, :, 2].mean() < 0.4
    assert 0.1 < fit._draws[-1, :, 3].mean() < 0.4

    assert 0.01 < fit._draws[-1, :, 0].var() < 0.02
    assert 0.01 < fit._draws[-1, :, 1].var() < 0.02
    assert 0.01 < fit._draws[-1, :, 2].var() < 0.02
    assert 0.01 < fit._draws[-1, :, 3].var() < 0.02


def test_bernoulli_to_frame(fit):
    df = fit.to_frame()
    assert 0.1 < df["theta"].mean() < 0.4
    assert 0.01 < df["theta"].var() < 0.02


def test_bernoulli_get_item(fit):
    assert -7.9 < fit["lp__"].mean() < -7.0
    assert 0.1 < fit["theta"].mean() < 0.4
    assert 0.01 < fit["theta"].var(ddof=1) < 0.02


def test_bernoulli_random_seed():
    fit = stan.build(program_code, data=data, random_seed=42).sample()
    assert -7.9 < fit["lp__"].mean() < -7.0
    assert 0.1 < fit["theta"].mean() < 0.4
    assert 0.01 < fit["theta"].var(ddof=1) < 0.02


def test_bernoulli_random_seed_same():
    fits = [stan.build(program_code, data=data, random_seed=42).sample() for _ in range(2)]
    assert np.allclose(*[fit["theta"] for fit in fits])


def test_bernoulli_random_seed_different(posterior):
    fits = [stan.build(program_code, data=data, random_seed=seed).sample() for seed in (1, 2)]
    assert not np.allclose(*[fit["theta"] for fit in fits])
