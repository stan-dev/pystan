import pytest

import stan

# program code designed to generate initialization failed exception
program_code = "parameters { real y; } model { y ~ uniform(1.9, 2); }" ""


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, random_seed=1)


def test_initialization_failed_exception(posterior):
    with pytest.raises(RuntimeError, match=r"Initialization between"):
        posterior.sample()
