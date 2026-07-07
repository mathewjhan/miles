"""HTTP smoke tests for MultiLoRAHTTPServer with a mock upstream (no Ray, no SGLang)."""

import asyncio
import json

import aiohttp
from aiohttp import web
import pytest

from miles.utils.multi_lora import (
    MultiLoRABackend,
    MultiLoRAHTTPServer,
    make_rid,
)


async def start_server(upstream_url: str, server_cls=MultiLoRAHTTPServer) -> tuple[MultiLoRABackend, MultiLoRAHTTPServer]:
    backend = MultiLoRABackend(4, upstream_url)
    srv = server_cls(backend)
    await backend.init()
    await srv.start()
    return backend, srv


async def stop_server(backend: MultiLoRABackend, srv: MultiLoRAHTTPServer) -> None:
    await srv.stop()
    await backend.close()


def _is_dummy(body: dict) -> bool:
    return body.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort"


async def _start_mock_upstream(delay: float = 0.0):
    async def handler(request):
        if delay:
            await asyncio.sleep(delay)
        body = await request.read()
        rid = json.loads(body).get("rid") if body else None
        return web.json_response({"text": "upstream-ok", "rid": rid})

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _post(session, url, payload):
    async with session.post(url, json=payload) as resp:
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_forward_active_returns_upstream():
    upstream_runner, upstream_url = await _start_mock_upstream()
    backend, srv = await start_server(upstream_url)
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"http://127.0.0.1:{srv.actual_port}/register_adapter", json={"name": "A"})
            rid = make_rid("A")
            status, body = await _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            assert status == 200
            assert body["text"] == "upstream-ok"
    finally:
        await stop_server(backend, srv)
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_deregister_mid_flight_dummies():
    upstream_runner, upstream_url = await _start_mock_upstream(delay=0.2)
    backend, srv = await start_server(upstream_url)
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"http://127.0.0.1:{srv.actual_port}/register_adapter", json={"name": "A"})
            rid = make_rid("A")
            task = asyncio.create_task(
                _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            )
            await asyncio.sleep(0.05)  # let it be forwarded/in-flight
            await s.post(f"http://127.0.0.1:{srv.actual_port}/deregister_adapter", json={"name": "A"})
            status, body = await task
            assert _is_dummy(body)
            assert body["text"] == ""
    finally:
        await stop_server(backend, srv)
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_deregister_aborts_in_flight_requests():
    aborts: list[dict] = []
    worker_urls: list[str] = []

    async def handler(request):
        if request.path == "/list_workers":
            return web.json_response({"urls": worker_urls})
        if request.path == "/abort_request":
            aborts.append(json.loads(await request.read()))
            return web.json_response({})
        await asyncio.sleep(0.2)  # keep /generate in flight
        return web.json_response({"text": "upstream-ok"})

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    upstream_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    worker_urls.append(upstream_url)

    backend, srv = await start_server(upstream_url)
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"http://127.0.0.1:{srv.actual_port}/register_adapter", json={"name": "A"})
            rid = make_rid("A")
            task = asyncio.create_task(
                _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            )
            await asyncio.sleep(0.05)  # let it be forwarded/in-flight
            await s.post(f"http://127.0.0.1:{srv.actual_port}/deregister_adapter", json={"name": "A"})
            await task
        assert aborts == [{"rid": rid}]
    finally:
        await stop_server(backend, srv)
        await runner.cleanup()


@pytest.mark.asyncio
async def test_custom_server_subclass_adds_routes():
    class CustomServer(MultiLoRAHTTPServer):
        def create_app(self):
            app = super().create_app()

            @app.middleware("http")
            async def tag_response(request, call_next):
                response = await call_next(request)
                response.headers["X-Custom-Server"] = "1"
                return response

            return app

        def add_routes(self, app):
            super().add_routes(app)
            app.get("/custom_status")(self.custom_status)

        async def custom_status(self):
            return {"custom": True, "active": self.backend.registry.active()}

    upstream_runner, upstream_url = await _start_mock_upstream()
    backend, srv = await start_server(upstream_url, server_cls=CustomServer)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{srv.actual_port}/custom_status") as resp:
                body = await resp.json()
                assert resp.headers.get("X-Custom-Server") == "1"
            assert body == {"custom": True, "active": {}}
    finally:
        await stop_server(backend, srv)
        await upstream_runner.cleanup()


@pytest.mark.asyncio
async def test_block_retired_adapter():
    upstream_runner, upstream_url = await _start_mock_upstream()
    backend, srv = await start_server(upstream_url)
    try:
        async with aiohttp.ClientSession() as s:
            rid = make_rid("A")  # never registered
            status, body = await _post(s, f"http://127.0.0.1:{srv.actual_port}/generate", {"rid": rid, "text": "hi"})
            assert _is_dummy(body)  # blocked, not forwarded
    finally:
        await stop_server(backend, srv)
        await upstream_runner.cleanup()
