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


def test_stanc_warning() -> None:
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
    }
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        stan.build(program_code=program_code)
    assert "assignment operator <- is deprecated in the Stan language" in buffer.getvalue()
