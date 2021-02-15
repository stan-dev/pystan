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
    file_set = set()
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_size = os.stat(file_path).st_size
            size += file_size
            file_set.add((file_path, file_size))
    return size, file_set


def test_fit_cache():
    """Test that a fit with a random seed set is cached."""

    cache_size_before, cache_files_before = file_usage(cache_path())
    # this fit is cacheable
    random_seed = random.randrange(1, 2 ** 16)
    normal_posterior = stan.build(program_code, random_seed=random_seed)
    normal_posterior.sample()
    cache_size_after, cache_files_after = file_usage(cache_path())
    assert cache_size_after > cache_size_before and cache_files_before.symmetric_difference(cache_files_after)

    # fit is now in cache
    cache_size_before, cache_files_before = file_usage(cache_path())
    normal_posterior.sample()
    cache_size_after, cache_files_after = file_usage(cache_path())
    assert cache_size_before == cache_size_after and not cache_files_before.symmetric_difference(cache_files_after)


def test_fit_cache_uncacheable():
    """Test that a fit with a random seed set is cached."""
    cache_size_before, cache_files_before = file_usage(cache_path())
    # this fit is NOT cacheable, should not be saved
    normal_posterior = stan.build(program_code)
    normal_posterior.sample()
    cache_size_after, cache_files_after = file_usage(cache_path())
    assert cache_size_before == cache_size_after and not cache_files_before.symmetric_difference(cache_files_after)
