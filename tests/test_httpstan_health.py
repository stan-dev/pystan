import pytest

import stan


@pytest.mark.asyncio
async def test_httpstan_health():
    async with stan.common.HttpstanClient() as client:
        resp = await client.get("/health")
        assert resp.status == 200
