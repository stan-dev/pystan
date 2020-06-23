import pytest
import aiohttp

import stan


@pytest.mark.asyncio
async def test_httpstan_health():
    async with stan.common.httpstan_server() as (host, port):
        async with aiohttp.request("GET", f"http://{host}:{port}/v1/health") as resp:
            assert resp.status == 200
