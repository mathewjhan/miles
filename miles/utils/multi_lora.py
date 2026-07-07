"""Multi-LoRA backend + HTTP proxy (no Ray, no torch).

``MultiLoRABackend`` is the shared brain (adapter registry, in-flight rid
tracking, gating, engine-facing abort). It has two thin transports: the
``MultiLoRAController`` Ray actor in ``miles.ray.multi_lora_controller`` and
the ``MultiLoRAHTTPServer`` defined here — both delegate every operation to
the backend. Everything here is plain asyncio + aiohttp, testable without Ray
or torch.

Correctness for adapter replacement: each rollout request carries
``rid = make_rid(adapter_name)``. The proxy blocks forwards for adapters no
longer active, dummies responses whose adapter was deregistered while the
request was in flight, and ``MultiLoRABackend.deregister`` aborts the
adapter's in-flight engine requests (by exact rid).
"""

import asyncio
import json
import logging
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from miles.utils.adapter_config import RegisteredAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterRegistry",
    "MultiLoRABackend",
    "MultiLoRAHTTPServer",
    "RID_SEPARATOR",
    "make_rid",
    "parse_adapter",
    "dummy_response_body",
    "extract_rid",
]


# Separator between adapter name and request uuid in rids. Must not appear in
# adapter names (enforced at registration) so that rid prefix matching in
# SGLang's abort_request cannot hit another adapter's requests.
RID_SEPARATOR = "::"


def make_rid(adapter_name: str) -> str:
    return f"{adapter_name}{RID_SEPARATOR}{uuid.uuid4().hex}"


def parse_adapter(rid: str) -> str:
    return rid.rsplit(RID_SEPARATOR, 1)[0]


class AdapterRegistry:
    """Adapter lifecycle: slot allocation, configs, per-adapter weight versions,
    and deferred cleanup of freed slots."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slots: dict[str, int] = {}
        self.configs: dict[str, Any] = {}
        self.pending_cleanup: dict[str, int] = {}
        self.slot_versions: dict[str, int] = {}

    def is_active(self, name: str) -> bool:
        return name in self.slots

    def register(self, name: str, config: Any) -> dict:
        if RID_SEPARATOR in name:
            raise ValueError(f"Adapter name '{name}' must not contain '{RID_SEPARATOR}'")
        if name in self.slots:
            raise ValueError(f"Adapter '{name}' already registered")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.slots[name] = slot
        self.configs[name] = config
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        slot = self.slots.pop(name, None)
        self.configs.pop(name, None)
        if slot is not None:
            self.pending_cleanup[name] = slot

    def free_slot(self, name: str) -> int:
        slot = self.pending_cleanup.pop(name, None)
        if slot is not None:
            self.free_slots.add(slot)
        self.slot_versions.pop(name, None)
        return slot if slot is not None else -1

    def increment_version(self, name: str) -> None:
        self.slot_versions[name] = self.slot_versions.get(name, 0) + 1

    def active_adapters(self) -> dict[str, RegisteredAdapter]:
        return {
            name: RegisteredAdapter(name, self.configs[name], slot, self.slot_versions.get(name, 0))
            for name, slot in self.slots.items()
        }

    def active(self) -> dict[str, int]:
        return dict(self.slots)


class MultiLoRABackend:
    """Shared brain behind the Ray actor and the HTTP server: adapter registry,
    in-flight rid tracking, request gating, and engine-facing abort. Both
    transports delegate here, so operations like deregister-with-abort have a
    single implementation.

    Subclass and override ``validate_adapter`` to reject registrations (raise
    ``ValueError``); wire the subclass via ``--multi-lora-backend-path``.
    """

    def __init__(self, max_adapters: int, upstream_url: str) -> None:
        self.registry = AdapterRegistry(max_adapters)
        self.in_flight: dict[str, str] = {}
        self.upstream_url = upstream_url.rstrip("/")
        self.client: aiohttp.ClientSession | None = None

    async def init(self) -> None:
        self.client = aiohttp.ClientSession()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def validate_adapter(self, name: str, config: Any) -> None:
        """Override to reject adapter registrations (raise ValueError)."""

    async def register(self, name: str, config: Any) -> dict:
        await self.validate_adapter(name, config)
        return self.registry.register(name, config)

    async def deregister(self, name: str) -> None:
        self.registry.deregister(name)
        await self.abort_adapter_requests(name)

    def on_forward(self, rid: str) -> bool:
        name = parse_adapter(rid)
        if not self.registry.is_active(name):
            return False
        self.in_flight[rid] = name
        return True

    def on_response(self, rid: str) -> bool:
        name = self.in_flight.pop(rid, None)
        if name is None:
            return True
        return not self.registry.is_active(name)

    async def worker_urls(self) -> list[str]:
        assert self.client is not None
        for endpoint, extract in (
            ("/list_workers", lambda body: body["urls"]),
            ("/workers", lambda body: [worker["url"] for worker in body["workers"]]),
        ):
            try:
                async with self.client.get(f"{self.upstream_url}{endpoint}") as resp:
                    if resp.status == 200:
                        return extract(await resp.json())
            except Exception:
                continue
        return []

    async def abort_adapter_requests(self, adapter_name: str) -> None:
        """Abort the adapter's in-flight requests on all workers, by exact rid:
        the engine drops aborts whose rid is not a known request, so a prefix
        rid would be silently ignored — and for the same reason, posting a rid
        to workers that don't own it is a harmless no-op."""
        rids = [rid for rid, name in self.in_flight.items() if name == adapter_name]
        if not rids:
            return
        urls = await self.worker_urls()

        async def abort(url: str, rid: str) -> None:
            async with self.client.post(f"{url}/abort_request", json={"rid": rid}) as resp:
                await resp.read()

        results = await asyncio.gather(*(abort(url, rid) for url in urls for rid in rids), return_exceptions=True)
        if failures := sum(isinstance(r, Exception) for r in results):
            logger.warning(f"Abort for adapter '{adapter_name}': {failures}/{len(results)} posts failed")


def dummy_response_body(rid: str) -> dict:
    return {
        "text": "",
        "meta_info": {"finish_reason": {"type": "abort"}},
        "rid": rid,
    }


def extract_rid(body: bytes) -> str | None:
    if not body:
        return None
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict):
        return obj.get("rid")
    return None


class MultiLoRAHTTPServer:
    """HTTP transport over a ``MultiLoRABackend``: control endpoints plus the
    catch-all proxy. Plain asyncio (no Ray), so it can be smoke-tested with a
    mock upstream.

    Subclass and override ``add_routes`` (calling ``super().add_routes(app)``) to
    expose custom endpoints; the catch-all proxy route is always registered after
    ``add_routes`` so custom routes take precedence.
    """

    def __init__(self, backend, host="127.0.0.1", port=0):
        self.backend = backend
        self.host = host
        self.port = port
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    @property
    def actual_port(self) -> int:
        return self.site._server.sockets[0].getsockname()[1] if self.site else self.port

    def create_app(self) -> web.Application:
        """Override to customize the application itself (e.g. middlewares for
        auth); ``start`` adds routes after this returns."""
        return web.Application()

    def add_routes(self, app: web.Application) -> None:
        app.router.add_post("/register_adapter", self.register_handler)
        app.router.add_post("/deregister_adapter", self.deregister_handler)
        app.router.add_get("/active_adapters", self.active_handler)

    async def start(self) -> None:
        app = self.create_app()
        self.add_routes(app)
        app.router.add_resource("/{tail:.*}").add_route("*", self.proxy_handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    async def register_handler(self, request):
        body = await request.json()
        result = await self.backend.register(body["name"], body.get("config"))
        return web.json_response({"ok": True, **result, "active": self.backend.registry.active()})

    async def deregister_handler(self, request):
        body = await request.json()
        await self.backend.deregister(body["name"])
        return web.json_response({"ok": True, "active": self.backend.registry.active()})

    async def active_handler(self, request):
        return web.json_response(self.backend.registry.active())

    async def proxy_handler(self, request):
        body = await request.read()
        rid = extract_rid(body)
        if rid is not None and not self.backend.on_forward(rid):
            return web.json_response(dummy_response_body(rid), status=200)
        url = f"{self.backend.upstream_url}/{request.match_info['tail']}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("content-length", "transfer-encoding", "host")}
        client = self.backend.client
        assert client is not None
        async with client.request(request.method, url, data=body, headers=headers) as upstream:
            content = await upstream.read()
            if rid is not None and self.backend.on_response(rid):
                return web.json_response(dummy_response_body(rid), status=200)
            out_headers = {k: v for k, v in upstream.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
            return web.Response(body=content, status=upstream.status, headers=out_headers)
