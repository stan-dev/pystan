"""CPU-intensive test, useful for testing progress bars."""
import stan

program_code = """
parameters {
  corr_matrix[30] Lambda;
}
"""


def test_corr_matrix_build():
    """Compile a simple model."""
    posterior = stan.build(program_code)
    assert posterior is not None


def test_corr_matrix_sample():
    """Sample from a simple model."""
    posterior = stan.build(program_code)
    fit = posterior.sample(num_chains=2, num_samples=50)
    df = fit.to_frame()
    assert len(df["Lambda.1.1"]) == 100
