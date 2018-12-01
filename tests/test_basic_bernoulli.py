import numpy as np
import pytest

import stan


program_code = "parameters {real y;} model {y ~ normal(0,1);}"


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


def test_bernoulli_sampling_error():
    bad_data = data.copy()
    del bad_data["N"]
    with pytest.raises(RuntimeError) as exc_info:
        stan.build(program_code, data=bad_data)
        assert "variable does not exist" in str(exc_info.value)


def test_bernoulli_sampling_thin(posterior):
    fit = posterior.sample(num_thin=2)
    assert fit.values.shape[1] == 500


def test_bernoulli_sampling_invalid_argument(posterior):
    with pytest.raises(TypeError) as exc_info:
        posterior.sample(num_thin=2.0)
        assert "only integer values allowed" in str(exc_info.value)


def test_bernoulli_sampling(fit):
    assert fit.num_samples == 1000
    assert fit.param_names == ("theta",)
    assert fit.num_chains == 4

    assert fit.values.ndim == 3
    assert fit.values.shape[1] == 1000
    assert fit.values.shape[2] == 4

    # for a fit with only one scalar parameter, it is the last one
    assert 0.1 < fit.values[-1, :, 0].mean() < 0.4
    assert 0.1 < fit.values[-1, :, 1].mean() < 0.4
    assert 0.1 < fit.values[-1, :, 2].mean() < 0.4
    assert 0.1 < fit.values[-1, :, 3].mean() < 0.4

    assert 0.01 < fit.values[-1, :, 0].var() < 0.02
    assert 0.01 < fit.values[-1, :, 1].var() < 0.02
    assert 0.01 < fit.values[-1, :, 2].var() < 0.02
    assert 0.01 < fit.values[-1, :, 3].var() < 0.02


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
    for i, fit in enumerate(fits):
        print(i, fit["theta"].ravel()[:10])
    assert not np.allclose(*[fit["theta"] for fit in fits])
