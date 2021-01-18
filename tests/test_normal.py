import stan


def test_normal_build():
    """Build (compile) a simple model."""
    program_code = "parameters {real y;} model {y ~ normal(0,1);}"
    posterior = stan.build(program_code)
    assert posterior is not None


def test_normal_sample():
    """Sample from normal distribution."""
    program_code = "parameters {real y;} model {y ~ normal(0, 0.0001);}"
    posterior = stan.build(program_code)
    assert posterior is not None
    fit = posterior.sample()
    offset = len(fit.sample_and_sampler_param_names)
    assert fit._draws.shape == (offset + 1, 1000, 1)  # 1 chain, n samples, 1 param
    df = fit.to_frame()
    assert (df["y"] == fit._draws[offset, :, :].ravel()).all()
    assert len(df["y"]) == 1000
    assert -0.01 < df["y"].mean() < 0.01
    assert -0.01 < df["y"].std() < 0.01


def test_normal_sample_chains():
    """Sample from normal distribution with more than one chain."""
    program_code = "parameters {real y;} model {y ~ normal(0,1);}"
    posterior = stan.build(program_code)
    assert posterior is not None
    fit = posterior.sample(num_chains=3)
    offset = len(fit.sample_and_sampler_param_names)
    assert fit._draws.shape == (offset + 1, 1000, 3)  # 1 param, n samples, 3 chains
    df = fit.to_frame()
    assert len(df["y"]) == 3000
    assert -5 < df["y"].mean() < 5


def test_normal_sample_args():
    """Sample from normal distribution with build arguments."""
    program_code = "parameters {real y;} model {y ~ normal(0,1);}"
    posterior = stan.build(program_code, random_seed=1)
    assert posterior is not None
    fit = posterior.sample(num_samples=350, num_thin=2)
    df = fit.to_frame()
    assert len(df["y"]) == 350 // 2
    assert -5 < df["y"].mean() < 5
