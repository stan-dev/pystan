"""Tests related to cached fits."""
import shutil
import pathlib
import random

import stan
import httpstan.cache

program_code = "parameters {real y;} model {y ~ normal(0,1);}"


def cache_path():
    return pathlib.Path(httpstan.cache.model_directory("models/abcdef")).parent


def test_fit_cache():
    """Test that a fit with a random seed set is cached."""

    cache_size_before = shutil.disk_usage(cache_path()).used
    print(cache_size_before)
    # this fit is cacheable
    random_seed = random.randrange(1, 2 ** 16)
    normal_posterior = stan.build(program_code, random_seed=random_seed)
    normal_posterior.sample()
    cache_size_after = shutil.disk_usage(cache_path()).used
    print(cache_size_after)
    assert cache_size_after > cache_size_before

    # fit is now in cache
    cache_size_before = shutil.disk_usage(cache_path()).used
    normal_posterior.sample()
    cache_size_after = shutil.disk_usage(cache_path()).used
    # allow for a 4096 byte difference (an empty directory takes 4K)
    assert abs(cache_size_before - cache_size_after) <= 4096


def test_fit_cache_uncacheable():
    """Test that a fit with a random seed set is cached."""
    cache_size_before = shutil.disk_usage(cache_path()).used
    # this fit is NOT cacheable, should not be saved
    normal_posterior = stan.build(program_code)
    normal_posterior.sample()
    cache_size_after = shutil.disk_usage(cache_path()).used
    # allow for a 4096 byte difference (an empty directory takes 4K)
    assert abs(cache_size_before - cache_size_after) <= 4096
