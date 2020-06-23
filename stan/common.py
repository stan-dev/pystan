"""Common routines"""
import contextlib
import socket

import aiohttp.web

import httpstan.app


def unused_tcp_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.asynccontextmanager
async def httpstan_server():
    """Manage starting and stopping the httpstan HTTP server."""
    host, port = "127.0.0.1", unused_tcp_port()
    app = httpstan.app.make_app()
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, host, port)
    await site.start()
    yield (host, port)
    await runner.cleanup()
