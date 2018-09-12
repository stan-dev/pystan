"""Common routines"""
import contextlib
import time
import typing

import requests

import httpstan.main


class ServerAddress(typing.NamedTuple):
    host: str
    port: int


@contextlib.contextmanager
def httpstan_server():
    """Manage starting and stopping an httpstan web gateway."""
    try:
        server = httpstan.main.Server()
        server.start()
        host, port = server.host, server.port
        retries = 10
        # if server is not ready (thread has not started)
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
        server.stop()
