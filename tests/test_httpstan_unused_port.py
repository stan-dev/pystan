import socket

import aiohttp
import pytest

import stan


@pytest.mark.asyncio
async def test_httpstan_port_conflict():
    s = socket.socket()
    try:
        s.bind(("", 8080))
        async with stan.common.httpstan_server() as (host, port):
            async with aiohttp.request("GET", f"http://{host}:{port}/v1/health") as resp:
                assert resp.status == 200
                assert port != 8080
    finally:
        s.close()
