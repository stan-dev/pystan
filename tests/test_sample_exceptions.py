"""Tests for sampling related exceptions."""
import pytest

import stan

# program code designed to generate initialization failed error
program_code = "parameters { real y; } model { y ~ uniform(10, 20); }" ""


def test_initialization_failed():
    posterior = stan.build(program_code, random_seed=1)
    with pytest.raises(RuntimeError, match=r"Initialization failed."):
        posterior.sample(num_chains=1)

    # run a second time, in case there are interactions with caching
    with pytest.raises(RuntimeError, match=r"Initialization failed."):
        posterior.sample(num_chains=1)
