"""Multi-LoRA backend + control-plane HTTP server (no Ray, no torch).

``MultiLoRABackend`` is the shared brain behind two thin transports: the
``MultiLoRAController`` Ray actor and the ``MultiLoRAHTTPServer`` here.
Control plane only — generation traffic goes to the router directly; on
deregister, one prefix abort (``rid = "{name}::"``) per worker reclaims the
adapter's in-flight requests.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from miles.utils.adapter_config import AdapterConfig, RegisteredAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterRegistry",
    "MultiLoRABackend",
    "MultiLoRAHTTPServer",
    "RID_SEPARATOR",
    "make_rid",
    "parse_adapter",
]


# Separator between adapter name and request uuid in rids. Must not appear in
# adapter names (enforced at registration) so that rid prefix matching in
# SGLang's abort_request cannot hit another adapter's requests.
RID_SEPARATOR = "::"

# Names become rid prefixes and filesystem path components (default save dirs),
# so restrict them to a path- and separator-safe alphabet.
VALID_ADAPTER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


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
    """Adapter lifecycle around three sets and per-slot monotonic versions.

    ``pending``: registered, weights not yet synced — invisible to generation.
    ``active``: weights synced at least once — sampleable. Promotion happens in
    ``record_weight_update``, i.e. exactly when a weight push made it true.
    ``cleanup``: deregistered, record retained until the trainer saves the final
    checkpoint and calls ``free_slot``.

    ``slot_versions`` count pushes to the slot and never reset, even across slot
    reuse (a new tenant continues where the previous one left off), so a
    (slot, version) pair never recurs: staleness deltas count this adapter's own
    pushes, and radix-cache salts can never collide with an earlier tenant's."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_versions: list[int] = [0] * max_adapters
        self.pending: dict[str, AdapterRecord] = {}
        self.active: dict[str, AdapterRecord] = {}
        # Deregistered but still serving until the next reconcile demotes them.
        self.retiring: dict[str, AdapterRecord] = {}
        # Being torn down (final ckpt, slot clear); removed by free_slot.
        self.cleanup: dict[str, AdapterRecord] = {}
        self.batch_adapters: dict[int, list[str]] = {}

    def all_records(self) -> dict[str, AdapterRecord]:
        return {**self.pending, **self.active, **self.retiring, **self.cleanup}

    def find(self, name: str) -> AdapterRecord | None:
        return self.all_records().get(name)

    def is_active(self, name: str) -> bool:
        return name in self.active or name in self.retiring

    def register(self, name: str, config: Any) -> dict:
        if not VALID_ADAPTER_NAME.match(name) or name in (".", ".."):
            raise ValueError(
                f"Adapter name '{name}' is invalid: use only letters, digits, '.', '_' and '-'"
            )
        if name in self.pending or name in self.active:
            raise ValueError(f"Adapter '{name}' already registered")
        if name in self.retiring or name in self.cleanup:
            raise ValueError(f"Adapter '{name}' is still cleaning up; retry shortly")
        if (save_dir := getattr(config, "save", None)) is not None:
            for record in self.all_records().values():
                other_save = getattr(record.config, "save", None)
                if other_save is not None and Path(other_save).resolve() == Path(save_dir).resolve():
                    raise ValueError(
                        f"Adapter '{name}' save dir '{save_dir}' is already used by adapter '{record.name}'"
                    )
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.pending[name] = AdapterRecord(name=name, slot=slot, config=config)
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        record = self.active.pop(name, None) or self.pending.pop(name, None)
        if record is not None:
            self.retiring[name] = record

    def retire_adapters(self) -> list[str]:
        demoted = sorted(self.retiring)
        for name in demoted:
            self.cleanup[name] = self.retiring.pop(name)
        return demoted

    def free_slot(self, name: str) -> int:
        record = self.cleanup.pop(name, None)
        if record is None:
            return -1
        self.free_slots.add(record.slot)
        return record.slot

    def record_weight_update(self, names: list[str]) -> None:
        """Weights for these adapters were pushed to the engines: bump their
        slot versions and promote any pending ones to active."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            self.slot_versions[record.slot] += 1
            if name in self.pending:
                self.active[name] = self.pending.pop(name)

    def record_batch_adapters(self, rollout_id: int, names: list[str]) -> None:
        self.batch_adapters[rollout_id] = list(names)
        while len(self.batch_adapters) > MAX_BATCH_RECORDS:
            self.batch_adapters.pop(next(iter(self.batch_adapters)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        trained = []
        for name in self.batch_adapters.pop(rollout_id, []):
            record = self.active.get(name) or self.retiring.get(name) or self.cleanup.get(name)
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
            version=self.slot_versions[record.slot],
            step=record.step,
        )

    def active_adapters(self) -> dict[str, RegisteredAdapter]:
        """The sampleable view: retiring adapters keep serving until retired."""
        return {name: self.view(record) for name, record in {**self.active, **self.retiring}.items()}

    def snapshot(self) -> dict:
        """Atomic per-phase view. The trainer keeps pending/active/retiring
        loaded and tears down cleanup."""
        return {
            "pending": {name: self.view(record) for name, record in self.pending.items()},
            "active": {name: self.view(record) for name, record in self.active.items()},
            "retiring": {name: self.view(record) for name, record in self.retiring.items()},
            "cleanup": list(self.cleanup),
        }



class MultiLoRABackend:
    """Shared brain behind the Ray actor and the HTTP server: adapter registry
    plus engine-facing abort (via the router's worker list). Subclass
    (``--multi-lora-backend-path``) and override ``validate_adapter`` to
    reject registrations."""

    def __init__(self, args: Any, router_url: str) -> None:
        self.args = args
        self.registry = AdapterRegistry(args.multi_lora_n_adapters)
        self.router_url = router_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None

    async def init(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def validate_adapter(self, name: str, config: Any) -> None:
        """Override to reject adapter registrations (raise ValueError)."""

    def resolve_save_dir(self, name: str, config: Any) -> Any:
        """Default a missing per-adapter save dir to {args.save}/adapters/{name};
        an explicit config.save is kept as-is."""
        if config is None or not hasattr(config, "save"):
            return config
        if config.save is not None:
            return config
        if getattr(self.args, "save", None) is None:
            raise ValueError(
                f"Adapter '{name}' has no save dir: set 'save' in the adapter config or pass --save"
            )
        return replace(config, save=Path(self.args.save) / "adapters" / name)

    async def register(self, name: str, config: Any) -> dict:
        await self.validate_adapter(name, config)
        config = self.resolve_save_dir(name, config)
        result = self.registry.register(name, config)
        resolved = getattr(config, "save", None)
        if resolved is not None:
            logger.info(f"Adapter '{name}' registered (slot {result['slot']}), checkpoints -> {resolved}")
        return result

    async def deregister(self, name: str) -> None:
        self.registry.deregister(name)

    async def retire_adapters(self) -> list[str]:
        names = self.registry.retire_adapters()
        for name in names:
            await self.abort_adapter_requests(name)
        return names

    async def worker_urls(self) -> list[str]:
        assert self.client is not None
        for endpoint, extract in (
            ("/list_workers", lambda body: body["urls"]),
            ("/workers", lambda body: [worker["url"] for worker in body["workers"]]),
        ):
            try:
                resp = await self.client.get(f"{self.router_url}{endpoint}")
                if resp.status_code == 200:
                    return extract(resp.json())
            except Exception:
                continue
        return []

    async def abort_adapter_requests(self, adapter_name: str) -> None:
        """Abort the adapter's in-flight requests: one prefix abort per worker."""
        prefix = f"{adapter_name}{RID_SEPARATOR}"
        urls = await self.worker_urls()
        if not urls:
            logger.warning(f"Abort for adapter '{adapter_name}': no workers discovered at {self.router_url}")
            return
        results = await asyncio.gather(
            *(
                self.client.post(f"{url}/abort_request", json={"rid": prefix, "prefix": True})
                for url in urls
            ),
            return_exceptions=True,
        )
        if failures := sum(isinstance(r, Exception) for r in results):
            logger.warning(f"Abort for adapter '{adapter_name}': {failures}/{len(results)} posts failed")


class RegisterRequest(BaseModel):
    name: str
    config: AdapterConfig | None = None


class DeregisterRequest(BaseModel):
    name: str


class MultiLoRAHTTPServer:
    """FastAPI control plane over a ``MultiLoRABackend``, served by an embedded
    uvicorn: register/deregister/active plus whatever subclasses add. No data
    plane — generation traffic goes to the inference router directly.

    Subclasses (``--multi-lora-http-server-path``) override ``add_routes`` for
    extra endpoints and ``create_app`` for middlewares (e.g. auth)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        self.backend = backend
        self.host = host
        self.api_port = api_port
        self.api_server: uvicorn.Server | None = None
        self.api_task: asyncio.Task | None = None

    @property
    def actual_api_port(self) -> int:
        if self.api_server is not None and self.api_server.started:
            return self.api_server.servers[0].sockets[0].getsockname()[1]
        return self.api_port

    def create_app(self) -> FastAPI:
        return FastAPI(title="Miles Multi-LoRA Controller")

    def add_routes(self, app: FastAPI) -> None:
        app.post("/register_adapter")(self.register_handler)
        app.post("/deregister_adapter")(self.deregister_handler)
        app.get("/active_adapters")(self.active_handler)

    async def start(self) -> None:
        app = self.create_app()
        self.add_routes(app)
        config = uvicorn.Config(app, host=self.host, port=self.api_port, log_level="warning", access_log=False)
        self.api_server = uvicorn.Server(config)
        self.api_task = asyncio.create_task(self.api_server.serve())
        while not self.api_server.started:
            if self.api_task.done():
                self.api_task.result()
                raise RuntimeError("uvicorn exited before startup completed")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self.api_server is not None:
            self.api_server.should_exit = True
            await self.api_task
        self.api_server = self.api_task = None

    async def register_handler(self, body: RegisterRequest):
        result = await self.backend.register(body.name, body.config)
        return {"ok": True, **result, "active": self.active_slots()}

    async def deregister_handler(self, body: DeregisterRequest):
        await self.backend.deregister(body.name)
        return {"ok": True, "active": self.active_slots()}

    def active_slots(self) -> dict[str, int]:
        return {name: adapter.slot for name, adapter in self.backend.registry.active_adapters().items()}

    async def active_handler(self):
        return {
            name: {"slot": adapter.slot, "version": adapter.version, "step": adapter.step}
            for name, adapter in self.backend.registry.active_adapters().items()
        }
