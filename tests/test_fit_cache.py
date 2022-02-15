"""Tests related to cached fits."""
import os
import pathlib
import random

import httpstan.cache

import stan

program_code = "parameters {real y;} model {y ~ normal(0,1);}"


def cache_path():
    return pathlib.Path(httpstan.cache.model_directory("models/abcdef")).parent


def file_usage(path):
    """Calculate the size used by the files in bytes."""
    size = 0
    for root, _, files in os.walk(path):
        for filename in files:
            size += os.stat(os.path.join(root, filename)).st_size
    return size


def test_fit_cache():
    """Test that a fit with a random seed set is cached."""

    cache_size_before = file_usage(cache_path())
    # this fit is cacheable
    random_seed = random.randrange(1, 2**16)
    normal_posterior = stan.build(program_code, random_seed=random_seed)
    normal_posterior.sample()
    cache_size_after = file_usage(cache_path())
    assert cache_size_after > cache_size_before

    # fit is now in cache
    cache_size_before = file_usage(cache_path())
    normal_posterior.sample()
    cache_size_after = file_usage(cache_path())
    assert cache_size_before == cache_size_after


def test_fit_cache_uncacheable():
    """Test that a fit with a random seed set is cached."""
    cache_size_before = file_usage(cache_path())
    # this fit is NOT cacheable, should not be saved
    normal_posterior = stan.build(program_code)
    normal_posterior.sample()
    cache_size_after = file_usage(cache_path())
    assert cache_size_before == cache_size_after
