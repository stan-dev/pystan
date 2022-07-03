"""Common routines"""
import socket
import typing

import aiohttp
import aiohttp.web
import httpstan.app
import simdjson


def unused_tcp_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class HTTPResponse(typing.NamedTuple):
    status: int
    content: bytes

    def json(self) -> dict:
        # mypy 0.961 complains that simdjson lacks a `loads`.
        return simdjson.loads(self.content)  # type: ignore


class HttpstanClient:
    """Manage starting and stopping the httpstan HTTP server."""

    async def __aenter__(self):
        app = httpstan.app.make_app()
        self.runner = aiohttp.web.AppRunner(app)
        await self.runner.setup()
        host, port = "127.0.0.1", unused_tcp_port()
        site = aiohttp.web.TCPSite(self.runner, host, port)
        await site.start()
        self.session = aiohttp.ClientSession()
        self.base_url = f"http://{host}:{port}/v1"
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()
        await self.runner.cleanup()

    async def get(self, path: str) -> HTTPResponse:
        async with self.session.get(f"{self.base_url}{path}") as resp:
            return HTTPResponse(status=resp.status, content=await resp.read())

    async def post(self, path: str, json: dict) -> HTTPResponse:
        async with self.session.post(f"{self.base_url}{path}", json=json) as resp:
            return HTTPResponse(status=resp.status, content=await resp.read())

    async def delete(self, path: str) -> HTTPResponse:
        async with self.session.delete(f"{self.base_url}{path}") as resp:
            return HTTPResponse(status=resp.status, content=await resp.read())
