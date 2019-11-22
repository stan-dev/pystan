"""Common routines"""
import asyncio
import contextlib
import threading
import time
import typing

import aiohttp.web
import requests

import httpstan.app


class ServerAddress(typing.NamedTuple):
    host: str
    port: int


@contextlib.contextmanager
def httpstan_server():
    """Manage starting and stopping an httpstan web gateway."""
    host, port = "127.0.0.1", 8080
    runner = aiohttp.web.AppRunner(httpstan.app.make_app())
    loop = asyncio.get_event_loop()

    try:
        # After dropping Python 3.6, use `asyncio.run`
        asyncio.get_event_loop().run_until_complete(runner.setup())
        site = aiohttp.web.TCPSite(runner, host, port)
        # After dropping Python 3.6, use `asyncio.run`
        asyncio.get_event_loop().run_until_complete(site.start())
        t = threading.Thread(target=loop.run_forever)
        # after this call, the event loop is running in thread which is not the main
        # thread. All interactions with the event loop must use thread-safe calls
        t.start()

        # wait until server is ready
        retries = 10
        for _ in range(retries):
            try:
                r = requests.get(f"http://{host}:{port}/v1/health", timeout=0.01)
            except requests.ConnectionError:
                time.sleep(0.01)
                continue
            except requests.Timeout:
                continue
            if r.status_code == 200:
                break
        else:
            raise RuntimeError("Could not communicate with httpstan server.")

        yield ServerAddress(host=host, port=port)

    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop)
        loop.call_soon_threadsafe(loop.stop)  # stops `run_forever`
        t.join(timeout=1)
