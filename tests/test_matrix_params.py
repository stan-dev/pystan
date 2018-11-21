import stan


def test_vector_params():
    """Sample from a program with vector params."""
    program_code = """
        parameters {
          vector[3] beta;
        }
        model {
          beta ~ normal(0, 1);
        }
    """
    posterior = stan.build(program_code)
    fit = posterior.sample()
    df = fit.to_frame()
    assert all(df.columns[-3:] == ["beta.1", "beta.2", "beta.3"])
    assert len(df["beta.1"]) > 100


def test_matrix_params_build():
    """Sample from a program with matrix-valued params."""
    program_code = """
        data {
          int<lower=2> K;
          int<lower=1> D;
        }
        parameters {
          matrix[K,D] beta;
        }
        model {
          for (k in 1:K)
            for (d in 1:D)
                beta[k,d] ~ normal(0, 1);
        }
    """
    data = {"K": 9, "D": 5}
    posterior = stan.build(program_code, data=data)
    assert posterior is not None


def test_matrix_params_sample():
    """Sample from a program with matrix-valued params."""
    program_code = """
        data {
          int<lower=2> K;
          int<lower=1> D;
        }
        parameters {
          matrix[K,D] beta;
        }
        model {
          for (k in 1:K)
            for (d in 1:D)
                beta[k,d] ~ normal(0, 1);
        }
    """
    data = {"K": 9, "D": 5}
    posterior = stan.build(program_code, data=data)
    fit = posterior.sample()
    df = fit.to_frame()
    assert len(df.columns) == len(fit.sample_and_sampler_param_names) + data["K"] * data["D"]
    assert len(df["beta.1.1"]) > 100
