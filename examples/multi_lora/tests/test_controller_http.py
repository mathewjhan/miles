"""HTTP smoke tests for MultiLoRAHTTPServer with a mock upstream (no Ray, no SGLang)."""

import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from types import SimpleNamespace

from miles.utils.multi_lora import MultiLoRABackend, MultiLoRAHTTPServer, make_rid


def _is_dummy(body: dict) -> bool:
    return body.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort"


class ControllerHarness:
    """Running controller (backend + both listeners) against a mock upstream
    that acts as router (/list_workers) and worker (/abort_request, echo)."""

    def __init__(self, session: aiohttp.ClientSession, backend: MultiLoRABackend, srv: MultiLoRAHTTPServer):
        self.session = session
        self.backend = backend
        self.srv = srv
        self.aborts: list[dict] = []

    @property
    def proxy_base(self) -> str:
        return f"http://127.0.0.1:{self.srv.actual_port}"

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.srv.actual_api_port}"

    async def api_post(self, path: str, payload: dict) -> tuple[int, dict]:
        async with self.session.post(f"{self.api_base}{path}", json=payload) as resp:
            return resp.status, await resp.json()

    async def api_get(self, path: str) -> tuple[int, dict, dict]:
        async with self.session.get(f"{self.api_base}{path}") as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, await resp.json(), headers

    async def proxy_post(self, path: str, payload: dict) -> tuple[int, dict]:
        async with self.session.post(f"{self.proxy_base}{path}", json=payload) as resp:
            return resp.status, await resp.json()

    async def register(self, name: str) -> tuple[int, dict]:
        status, body = await self.api_post("/register_adapter", {"name": name})
        # Registered adapters start pending; a weight push promotes them.
        self.backend.registry.record_weight_update([name])
        return status, body

    async def deregister(self, name: str) -> tuple[int, dict]:
        return await self.api_post("/deregister_adapter", {"name": name})

    async def generate(self, rid: str) -> tuple[int, dict]:
        return await self.proxy_post("/generate", {"rid": rid, "text": "hi"})


@asynccontextmanager
async def running_controller(delay: float = 0.0, server_cls=MultiLoRAHTTPServer):
    upstream_url = ""
    harness: ControllerHarness | None = None

    async def upstream_handler(request):
        if request.path == "/list_workers":
            return web.json_response({"urls": [upstream_url]})
        if request.path == "/abort_request":
            harness.aborts.append(json.loads(await request.read()))
            return web.json_response({})
        if delay:
            await asyncio.sleep(delay)
        body = await request.read()
        rid = json.loads(body).get("rid") if body else None
        return web.json_response({"text": "upstream-ok", "rid": rid})

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", upstream_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    upstream_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    backend = MultiLoRABackend(SimpleNamespace(multi_lora_n_adapters=4, save=None), upstream_url)
    srv = server_cls(backend)
    await backend.init()
    await srv.start()
    try:
        async with aiohttp.ClientSession() as session:
            harness = ControllerHarness(session, backend, srv)
            yield harness
    finally:
        await srv.stop()
        await backend.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_forward_active_returns_upstream():
    async with running_controller() as ctl:
        await ctl.register("A")
        status, body = await ctl.generate(make_rid("A"))
        assert status == 200
        assert body["text"] == "upstream-ok"


@pytest.mark.asyncio
async def test_block_retired_adapter():
    async with running_controller() as ctl:
        _, body = await ctl.generate(make_rid("A"))  # never registered
        assert _is_dummy(body)  # blocked, not forwarded


@pytest.mark.asyncio
async def test_deregister_mid_flight_dummies():
    async with running_controller(delay=0.2) as ctl:
        await ctl.register("A")
        task = asyncio.create_task(ctl.generate(make_rid("A")))
        await asyncio.sleep(0.05)  # let it be forwarded/in-flight
        await ctl.deregister("A")
        _, body = await task
        assert _is_dummy(body)
        assert body["text"] == ""


@pytest.mark.asyncio
async def test_deregister_aborts_in_flight_requests():
    async with running_controller(delay=0.2) as ctl:
        await ctl.register("A")
        rid = make_rid("A")
        task = asyncio.create_task(ctl.generate(rid))
        await asyncio.sleep(0.05)  # let it be forwarded/in-flight
        await ctl.deregister("A")
        await task
        assert ctl.aborts == [{"rid": rid}]


@pytest.mark.asyncio
async def test_control_routes_only_on_api_listener():
    """A control route hitting the proxy port is forwarded upstream, not handled."""
    async with running_controller() as ctl:
        assert ctl.srv.actual_port != ctl.srv.actual_api_port
        _, body = await ctl.proxy_post("/register_adapter", {"name": "A"})
        assert body.get("text") == "upstream-ok"
        assert ctl.backend.registry.active_adapters() == {}


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
            return {"custom": True, "active": self.active_slots()}

    async with running_controller(server_cls=CustomServer) as ctl:
        _, body, headers = await ctl.api_get("/custom_status")
        assert headers.get("x-custom-server") == "1"
        assert body == {"custom": True, "active": {}}
