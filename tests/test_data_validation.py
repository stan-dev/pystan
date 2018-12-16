import json

import numpy as np
import pandas as pd
import pytest

import stan.model


def test_ensure_json():
    data = {"K": 3}
    assert stan.model._ensure_json_serializable(data) == data
    data = {"K": 3, "x": [3, 4, 5]}
    assert stan.model._ensure_json_serializable(data) == data

    class DummyClass:
        pass

    data = {"K": DummyClass(), "x": [3, 4, 5]}
    with pytest.raises(TypeError, match=r"Value associated with variable `K`"):
        stan.model._ensure_json_serializable(data)


def test_ensure_json_numpy():
    data = {"K": 3, "x": np.array([3, 4, 5])}
    expected = {"K": 3, "x": [3, 4, 5]}
    assert stan.model._ensure_json_serializable(data) == expected
    assert json.dumps(stan.model._ensure_json_serializable(data))


def test_ensure_json_pandas():
    data = {"K": 3, "x": pd.Series([3, 4, 5])}
    expected = {"K": 3, "x": [3, 4, 5]}
    assert stan.model._ensure_json_serializable(data) == expected
    assert json.dumps(stan.model._ensure_json_serializable(data))
