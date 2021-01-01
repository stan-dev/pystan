import contextlib
import io

import pytest

import stan

# program code designed to generate initialization failed messages
program_code = "parameters { real y; } model { y ~ uniform(1.9, 2); }" ""


@pytest.fixture(scope="module")
def posterior():
    return stan.build(program_code, random_seed=1)


def test_logger_messages_present(posterior):
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        posterior.sample()
    s = f.getvalue()
    assert "Rejecting initial value:" in s
