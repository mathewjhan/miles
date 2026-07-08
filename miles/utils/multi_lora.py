"""Multi-LoRA backend + HTTP proxy (no Ray, no torch).

``MultiLoRABackend`` is the shared brain (adapter registry, in-flight rid
tracking, gating, engine-facing abort). It has two thin transports: the
``MultiLoRAController`` Ray actor in ``miles.ray.multi_lora_controller`` and
the ``MultiLoRAHTTPServer`` defined here — both delegate every operation to
the backend. FastAPI + httpx + uvicorn (same stack as the miles router),
testable without Ray or torch.

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
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

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


@dataclass
class AdapterRecord:
    name: str
    slot: int
    config: Any
    step: int = 0


# Retained batch records, bounding leakage from cycles that crash before
# mark_batch_trained.
MAX_BATCH_RECORDS = 16


class AdapterRegistry:
    """Adapter lifecycle around three sets and per-slot monotonic counters.

    ``pending``: registered, weights not yet synced — invisible to generation.
    ``active``: weights synced at least once — sampleable. Promotion happens in
    ``record_weight_update``, i.e. exactly when a weight push made it true.
    ``cleanup``: deregistered, record retained until the trainer saves the final
    checkpoint and calls ``free_slot``.

    ``slot_counters`` never reset, even across slot reuse, so a (slot, counter)
    pair never recurs: staleness deltas count this adapter's own pushes, and
    radix-cache salts can never collide with an earlier tenant's."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_counters: list[int] = [0] * max_adapters
        self.pending: dict[str, AdapterRecord] = {}
        self.active: dict[str, AdapterRecord] = {}
        self.cleanup: dict[str, AdapterRecord] = {}
        self.batch_adapters: dict[int, list[str]] = {}

    def find(self, name: str) -> AdapterRecord | None:
        return self.active.get(name) or self.pending.get(name) or self.cleanup.get(name)

    def is_active(self, name: str) -> bool:
        return name in self.active

    def register(self, name: str, config: Any) -> dict:
        if RID_SEPARATOR in name:
            raise ValueError(f"Adapter name '{name}' must not contain '{RID_SEPARATOR}'")
        if name in self.pending or name in self.active:
            raise ValueError(f"Adapter '{name}' already registered")
        if name in self.cleanup:
            raise ValueError(f"Adapter '{name}' is still cleaning up; retry shortly")
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.pending[name] = AdapterRecord(name=name, slot=slot, config=config)
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        record = self.active.pop(name, None) or self.pending.pop(name, None)
        if record is not None:
            self.cleanup[name] = record

    def free_slot(self, name: str) -> int:
        record = self.cleanup.pop(name, None)
        if record is None:
            return -1
        self.free_slots.add(record.slot)
        return record.slot

    def record_weight_update(self, names: list[str]) -> None:
        """Weights for these adapters were pushed to the engines: bump their
        slot counters and promote any pending ones to active."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            self.slot_counters[record.slot] += 1
            if name in self.pending:
                self.active[name] = self.pending.pop(name)

    def record_batch_adapters(self, rollout_id: int, names: list[str]) -> None:
        self.batch_adapters[rollout_id] = list(names)
        while len(self.batch_adapters) > MAX_BATCH_RECORDS:
            self.batch_adapters.pop(next(iter(self.batch_adapters)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        trained = []
        for name in self.batch_adapters.pop(rollout_id, []):
            record = self.active.get(name) or self.cleanup.get(name)
            if record is not None:
                record.step += 1
                trained.append(name)
        return trained

    def set_step(self, name: str, step: int) -> None:
        if (record := self.find(name)) is not None:
            record.step = step

    def step_count(self, name: str) -> int:
        record = self.find(name)
        return record.step if record is not None else 0

    def view(self, record: AdapterRecord) -> RegisteredAdapter:
        return RegisteredAdapter(
            name=record.name,
            config=record.config,
            slot=record.slot,
            version=self.slot_counters[record.slot],
            step=record.step,
        )

    def active_adapters(self) -> dict[str, RegisteredAdapter]:
        return {name: self.view(record) for name, record in self.active.items()}

    def snapshot(self) -> dict:
        """Atomic view of all three sets, in the registry's own vocabulary.
        The trainer loads pending + active and cleans up cleanup."""
        return {
            "pending": {name: self.view(record) for name, record in self.pending.items()},
            "active": {name: self.view(record) for name, record in self.active.items()},
            "cleanup": list(self.cleanup),
        }



class MultiLoRABackend:
    """Shared brain behind the Ray actor and the HTTP server: adapter registry,
    in-flight rid tracking, request gating, and engine-facing abort. Subclass
    (``--multi-lora-backend-path``) and override ``validate_adapter`` to
    reject registrations."""

    def __init__(self, max_adapters: int, upstream_url: str) -> None:
        self.registry = AdapterRegistry(max_adapters)
        self.in_flight: dict[str, str] = {}
        self.upstream_url = upstream_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None

    async def init(self) -> None:
        # No timeout: proxied generate requests run for minutes.
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024),
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
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
                resp = await self.client.get(f"{self.upstream_url}{endpoint}")
                if resp.status_code == 200:
                    return extract(resp.json())
            except Exception:
                continue
        return []

    async def abort_adapter_requests(self, adapter_name: str) -> None:
        """Abort by exact rid on every worker: the engine drops aborts for
        unknown rids, so prefix rids are ignored and wrong-worker posts are
        harmless no-ops."""
        rids = [rid for rid, name in self.in_flight.items() if name == adapter_name]
        if not rids:
            return
        urls = await self.worker_urls()
        results = await asyncio.gather(
            *(self.client.post(f"{url}/abort_request", json={"rid": rid}) for url in urls for rid in rids),
            return_exceptions=True,
        )
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
    """FastAPI transport over a ``MultiLoRABackend``, served by embedded
    uvicorns on two listeners with different audiences:

    * proxy listener (``port``): the catch-all data-plane proxy that rollout
      requests flow through. Cluster-internal; never expose it.
    * api listener (``api_port``): the control plane (register/deregister/
      active + subclass routes). The only listener that should be exposed.

    Subclasses override ``add_routes`` and ``create_app`` (e.g. middlewares) —
    both scoped to the api app, so custom routes and auth can never shadow or
    leak the proxy."""

    def __init__(self, backend, host="127.0.0.1", port=0, api_port=0):
        self.backend = backend
        self.host = host
        self.port = port
        self.api_port = api_port
        self.proxy_server: uvicorn.Server | None = None
        self.proxy_task: asyncio.Task | None = None
        self.api_server: uvicorn.Server | None = None
        self.api_task: asyncio.Task | None = None

    @staticmethod
    def _bound_port(server: uvicorn.Server | None, configured: int) -> int:
        if server is not None and server.started:
            return server.servers[0].sockets[0].getsockname()[1]
        return configured

    @property
    def actual_port(self) -> int:
        return self._bound_port(self.proxy_server, self.port)

    @property
    def actual_api_port(self) -> int:
        return self._bound_port(self.api_server, self.api_port)

    def create_app(self) -> FastAPI:
        return FastAPI(title="Miles Multi-LoRA Controller")

    def add_routes(self, app: FastAPI) -> None:
        app.post("/register_adapter")(self.register_handler)
        app.post("/deregister_adapter")(self.deregister_handler)
        app.get("/active_adapters")(self.active_handler)

    def create_proxy_app(self) -> FastAPI:
        app = FastAPI(title="Miles Multi-LoRA Proxy")
        app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy_handler)
        return app

    async def _serve(self, app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task]:
        config = uvicorn.Config(app, host=self.host, port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("uvicorn exited before startup completed")
            await asyncio.sleep(0.01)
        return server, task

    async def start(self) -> None:
        self.proxy_server, self.proxy_task = await self._serve(self.create_proxy_app(), self.port)
        api_app = self.create_app()
        self.add_routes(api_app)
        self.api_server, self.api_task = await self._serve(api_app, self.api_port)

    async def stop(self) -> None:
        for server, task in ((self.api_server, self.api_task), (self.proxy_server, self.proxy_task)):
            if server is not None:
                server.should_exit = True
                await task
        self.proxy_server = self.proxy_task = None
        self.api_server = self.api_task = None

    async def register_handler(self, request: Request):
        body = await request.json()
        result = await self.backend.register(body["name"], body.get("config"))
        return {"ok": True, **result, "active": self.active_slots()}

    async def deregister_handler(self, request: Request):
        body = await request.json()
        await self.backend.deregister(body["name"])
        return {"ok": True, "active": self.active_slots()}

    def active_slots(self) -> dict[str, int]:
        return {name: adapter.slot for name, adapter in self.backend.registry.active_adapters().items()}

    async def active_handler(self):
        return self.active_slots()

    async def proxy_handler(self, request: Request):
        body = await request.body()
        rid = extract_rid(body)
        if rid is not None and not self.backend.on_forward(rid):
            return JSONResponse(dummy_response_body(rid))
        url = f"{self.backend.upstream_url}/{request.path_params['path']}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("content-length", "transfer-encoding", "host")}
        client = self.backend.client
        assert client is not None
        upstream = await client.request(request.method, url, content=body, headers=headers)
        if rid is not None and self.backend.on_response(rid):
            return JSONResponse(dummy_response_body(rid))
        out_headers = {k: v for k, v in upstream.headers.items()
                       if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
        return Response(content=upstream.content, status_code=upstream.status_code, headers=out_headers)
