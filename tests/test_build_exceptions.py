"""Tests for build related exceptions."""
import pytest

import stan


def test_semantic_error():
    # wrong number of arguments to `uniform`
    program_code = "parameters { real y; } model { y ~ uniform(10, 20, 30); }" ""
    with pytest.raises(ValueError, match=r"Semantic error"):
        stan.build(program_code, random_seed=1)


def test_syntax_error():
    program_code = "parameters { real y; } model { y ~ uniform(10| 20); }" ""
    with pytest.raises(ValueError, match=r"Syntax error"):
        stan.build(program_code, random_seed=1)
