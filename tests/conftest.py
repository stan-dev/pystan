"""pytest configuration for all tests."""
import multiprocessing

import pytest

import httpstan.main


@pytest.fixture(scope='session')
def host():
    """Host for server to listen on."""
    return '127.0.0.1'


@pytest.fixture(scope='session')
def port():
    """Port for server to listen on."""
    return 8080


@pytest.fixture(scope='session', autouse=True)  # FIXME: autouse is lazy, avoid
def httpstan_server(request, host, port):
    """Run httpstan server in the background."""

    import asyncio # FIXME: avoid using this directly
    import aiohttp  # FIXME: avoid using this directly

    def run_app(host, port):
        # setup server
        loop = asyncio.get_event_loop()
        app = httpstan.main.make_app(loop)
        aiohttp.web.run_app(app, host=host, port=port)

    th = multiprocessing.Process(target=run_app, args=(host, port), daemon=True)
    th.start()
    import time
    time.sleep(2)
    return th
