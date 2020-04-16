import numpy as np

import stan

program_code = """
    data {
      real x;
    }
    parameters {
      real mu;
    }
    model {
      x ~ normal(mu,1);
    }
"""
data = {"x": 2}


def test_program():
    """Compile the program."""
    posterior = stan.build(program_code, data=data)
    assert posterior is not None


def test_user_init_same_initial_values():
    posterior = stan.build(program_code, data=data, random_seed=2)

    mu1 = posterior.sample(num_chains=1, num_samples=10)["mu"].ravel()
    mu2 = posterior.sample(num_chains=1, num_samples=10)["mu"].ravel()
    assert mu1[0] == mu2[0]
    np.testing.assert_array_equal(mu1, mu2)

    mu3 = posterior.sample(num_chains=1, num_samples=10, init=[{"mu": -4}])["mu"].ravel()
    mu4 = posterior.sample(num_chains=1, num_samples=10, init=[{"mu": -4}])["mu"].ravel()
    assert mu3[0] == mu4[0]
    np.testing.assert_array_equal(mu3, mu4)
    assert mu1[0] != mu3[0]


def test_user_init_different_initial_values():
    posterior = stan.build(program_code, data=data, random_seed=2)

    mu1 = posterior.sample(num_chains=1, num_samples=10, init=[{"mu": 3}])["mu"].ravel()
    mu2 = posterior.sample(num_chains=1, num_samples=10, init=[{"mu": 4}])["mu"].ravel()
    assert mu1[0] != mu2[0]
