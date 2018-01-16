import asyncio
import json

import aiohttp

import httpstan.main


DEFAULT_HEADERS = {'content-type': 'application/json'}


async def _start_httpstan(host, port, loop):
    """Returns srv, handler, app."""
    app = httpstan.main.make_app(loop=loop)
    handler = app.make_handler()
    srv = await loop.create_server(handler, host, port)
    await app.startup()
    return srv, handler, app


async def _shutdown_httpstan(srv, handler, app):
    """Graceful shutdown."""
    srv.close()
    await srv.wait_closed()
    await app.shutdown()
    await handler.shutdown(0.01)
    await app.cleanup()


async def post(host, port, path, payload):
    """Make POST request, blocking until complete."""
    loop = asyncio.get_event_loop()
    srv, handler, app = await _start_httpstan(host, port, loop)
    
    try:
        async with aiohttp.ClientSession() as session:
            assert path.startswith('/')
            url = f'http://{host}:{port}{path}'
            async with session.post(url, data=json.dumps(payload), headers=DEFAULT_HEADERS) as resp:
                assert resp.status == 200, await resp.text()
                payload_response = await resp.json()
    finally:
        await _shutdown_httpstan(srv, handler, app)
    return payload_response


async def post_aiter(host, port, path, payload):
    """Make POST request, response is streaming ndjson."""
    loop = asyncio.get_event_loop()
    srv, handler, app = await _start_httpstan(host, port, loop)

    try:
        async with aiohttp.ClientSession() as session:
            assert path.startswith('/')
            url = f'http://{host}:{port}{path}'
            async with session.post(url, data=json.dumps(payload), headers=DEFAULT_HEADERS) as resp:
                assert resp.status == 200, await resp.text()
                async for chunk in resp.content:
                    yield json.loads(chunk)
    finally:
        await _shutdown_httpstan(srv, handler, app)
