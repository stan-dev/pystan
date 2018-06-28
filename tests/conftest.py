"""pytest configuration for all tests."""
import pytest


@pytest.fixture(scope="session")
def host():
    """Host for server to listen on."""
    return "127.0.0.1"


@pytest.fixture(scope="session")
def port():
    """Port for server to listen on."""
    return 8080
