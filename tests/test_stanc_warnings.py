"""Test that stanc warnings are visible."""
import contextlib
import io

import stan


def test_stanc_no_warning() -> None:
    """No warnings."""
    program_code = "parameters {real y;} model {y ~ normal(0,1);}"
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        stan.build(program_code=program_code)
    assert "warning" not in buffer.getvalue().lower()


def test_stanc_unused_warning() -> None:
    """Test that stanc warning is shown to user."""
    program_code = """
    parameters {
    real y;
    }
    model {
    real x;
    x = 5;
    }
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        stan.build(program_code=program_code)
    assert "The parameter y was declared but was not used in the density" in buffer.getvalue()


def test_stanc_assignment_warning() -> None:
    """Test that stanc warning is shown to user."""
    # stanc prints warning:
    # assignment operator <- is deprecated in the Stan language; use = instead.
    program_code = """
    parameters {
    real y;
    }
    model {
    real x;
    x <- 5;
    y ~ normal(0,1);
    }
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        stan.build(program_code=program_code)
    assert "operator <- is deprecated in the Stan language and will be removed" in buffer.getvalue(), buffer.getvalue()
