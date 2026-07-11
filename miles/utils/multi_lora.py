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
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from miles.utils.adapter_config import AdapterConfig, RegisteredAdapter, parse_adapter_yaml

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterRegistry",
    "AdapterState",
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


class AdapterState(str, Enum):
    PENDING = "PENDING"  # registered, weights not yet synced
    ACTIVE = "ACTIVE"  # weights synced at least once — sampleable
    RETIRING = "RETIRING"  # deregistered, still serving until the next reconcile
    CLEANUP = "CLEANUP"  # being torn down (final ckpt, slot clear)
    COMPLETED = "COMPLETED"  # slot freed; record retained for status queries


# States that hold a slot (everything but COMPLETED).
LIVE_STATES = (
    AdapterState.PENDING,
    AdapterState.ACTIVE,
    AdapterState.RETIRING,
    AdapterState.CLEANUP,
)


@dataclass
class AdapterRecord:
    name: str
    slot: int
    config: Any
    step: int = 0
    state: AdapterState = AdapterState.PENDING


# Retained batch records, bounding leakage from cycles that crash before
# mark_batch_trained.
MAX_BATCH_RECORDS = 16

# Completed adapters retained for status queries (LRU: reads refresh recency,
# so records being polled are not evicted under churn).
MAX_COMPLETED_RECORDS = 1024


class AdapterRegistry:
    """Adapter lifecycle: one record per name, each carrying its
    ``AdapterState``; every per-state view is derived by filtering.
    Transitions are single state assignments, so a record is in exactly one
    state by construction. Promotion (PENDING -> ACTIVE) happens in
    ``record_weight_update``, i.e. exactly when a weight push made it true.

    ``slot_versions`` count pushes to the slot and never reset, even across slot
    reuse (a new tenant continues where the previous one left off), so a
    (slot, version) pair never recurs: staleness deltas count this adapter's own
    pushes, and radix-cache salts can never collide with an earlier tenant's."""

    def __init__(self, max_adapters: int) -> None:
        self.max_adapters = max_adapters
        self.free_slots: set[int] = set(range(max_adapters))
        self.slot_versions: list[int] = [0] * max_adapters
        self.records: dict[str, AdapterRecord] = {}
        self.batch_adapters: dict[int, list[str]] = {}

    def in_state(self, *states: AdapterState) -> dict[str, AdapterRecord]:
        return {name: r for name, r in self.records.items() if r.state in states}

    def find(self, name: str) -> AdapterRecord | None:
        """The slot-holding record for a name (COMPLETED records excluded)."""
        record = self.records.get(name)
        return record if record is not None and record.state in LIVE_STATES else None

    def is_active(self, name: str) -> bool:
        record = self.records.get(name)
        return record is not None and record.state in (AdapterState.ACTIVE, AdapterState.RETIRING)

    def register(self, name: str, config: Any) -> dict:
        if not VALID_ADAPTER_NAME.match(name) or name in (".", ".."):
            raise ValueError(
                f"Adapter name '{name}' is invalid: use only letters, digits, '.', '_' and '-'"
            )
        if (existing := self.records.get(name)) is not None:
            if existing.state in (AdapterState.PENDING, AdapterState.ACTIVE):
                raise ValueError(f"Adapter '{name}' already registered")
            if existing.state in (AdapterState.RETIRING, AdapterState.CLEANUP):
                raise ValueError(f"Adapter '{name}' is still cleaning up; retry shortly")
        if (save_dir := getattr(config, "save", None)) is not None:
            for record in self.in_state(*LIVE_STATES).values():
                other_save = getattr(record.config, "save", None)
                if other_save is not None and Path(other_save).resolve() == Path(save_dir).resolve():
                    raise ValueError(
                        f"Adapter '{name}' save dir '{save_dir}' is already used by adapter '{record.name}'"
                    )
        if not self.free_slots:
            raise RuntimeError(f"No free adapter slots (max {self.max_adapters})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)
        self.records.pop(name, None)  # drop a stale COMPLETED record
        self.records[name] = AdapterRecord(name=name, slot=slot, config=config)
        return {"name": name, "slot": slot}

    def deregister(self, name: str) -> None:
        record = self.records.get(name)
        if record is not None and record.state in (AdapterState.PENDING, AdapterState.ACTIVE):
            record.state = AdapterState.RETIRING

    def retire_adapters(self) -> list[str]:
        retired = sorted(self.in_state(AdapterState.RETIRING))
        for name in retired:
            self.records[name].state = AdapterState.CLEANUP
        return retired

    def free_slot(self, name: str) -> int:
        record = self.records.get(name)
        if record is None or record.state is not AdapterState.CLEANUP:
            return -1
        self.free_slots.add(record.slot)
        record.state = AdapterState.COMPLETED
        # Reinsert so COMPLETED records order oldest-first for LRU eviction.
        self.records[name] = self.records.pop(name)
        completed = self.in_state(AdapterState.COMPLETED)
        for oldest in list(completed)[: len(completed) - MAX_COMPLETED_RECORDS]:
            self.records.pop(oldest)
        return record.slot

    def adapter_state(self, name: str) -> AdapterState | None:
        record = self.records.get(name)
        if record is None:
            return None
        if record.state is AdapterState.COMPLETED:
            # LRU touch: keep records that are still being polled.
            self.records[name] = self.records.pop(name)
        return record.state

    def record_weight_update(self, names: list[str]) -> None:
        """Weights for these adapters were pushed to the engines: bump their
        slot versions and promote any pending ones to active."""
        for name in names:
            record = self.find(name)
            if record is None:
                continue
            self.slot_versions[record.slot] += 1
            if record.state is AdapterState.PENDING:
                record.state = AdapterState.ACTIVE

    def record_batch_adapters(self, rollout_id: int, names: list[str]) -> None:
        self.batch_adapters[rollout_id] = list(names)
        while len(self.batch_adapters) > MAX_BATCH_RECORDS:
            self.batch_adapters.pop(next(iter(self.batch_adapters)))

    def mark_batch_trained(self, rollout_id: int) -> list[str]:
        trained = []
        for name in self.batch_adapters.pop(rollout_id, []):
            record = self.records.get(name)
            if record is not None and record.state in (
                AdapterState.ACTIVE,
                AdapterState.RETIRING,
                AdapterState.CLEANUP,
            ):
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
        return {
            name: self.view(record)
            for name, record in self.in_state(AdapterState.ACTIVE, AdapterState.RETIRING).items()
        }

    def snapshot(self) -> dict:
        """Atomic per-state view. The trainer keeps pending/active/retiring
        loaded and tears down cleanup."""

        def views(state: AdapterState) -> dict[str, RegisteredAdapter]:
            return {name: self.view(record) for name, record in self.in_state(state).items()}

        return {
            "pending": views(AdapterState.PENDING),
            "active": views(AdapterState.ACTIVE),
            "retiring": views(AdapterState.RETIRING),
            "cleanup": list(self.in_state(AdapterState.CLEANUP)),
            "completed": list(self.in_state(AdapterState.COMPLETED)),
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


class RegisterAdapterRequest(BaseModel):
    """Exactly one of ``config`` (inline) or ``yaml_path`` must be set."""

    name: str
    config: AdapterConfig | None = None
    yaml_path: str | None = None


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
        app = FastAPI(title="Miles Multi-LoRA Controller")

        @app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            return JSONResponse({"detail": str(exc)}, status_code=400)

        @app.exception_handler(RuntimeError)
        async def runtime_error_handler(request: Request, exc: RuntimeError):
            status = 409 if "No free adapter slots" in str(exc) else 500
            return JSONResponse({"detail": str(exc)}, status_code=status)

        return app

    def add_routes(self, app: FastAPI) -> None:
        app.get("/health")(self.health)
        app.get("/adapters")(self.list_adapters)
        # Registered before /adapters/{name} so "state" is not read as a name.
        app.get("/adapters/state")(self.adapter_states)
        app.get("/adapters/{name}")(self.get_adapter)
        app.post("/adapters")(self.register_adapter)
        app.delete("/adapters/{name}")(self.deregister_adapter)

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

    async def health(self) -> dict:
        return {"status": "healthy"}

    def adapter_statuses(self) -> list[dict]:
        """Flatten each RegisteredAdapter view plus its lifecycle state into
        the wire shape (config fields at the top level)."""
        registry = self.backend.registry
        statuses = []
        for record in registry.records.values():
            flat = asdict(registry.view(record))
            flat |= flat.pop("config")
            flat["save"] = str(flat["save"])
            flat["state"] = record.state
            if record.state is AdapterState.COMPLETED:
                # The slot may already host another tenant; its counter is
                # not this adapter's version.
                flat["version"] = None
            statuses.append(flat)
        return statuses

    async def list_adapters(self) -> dict:
        return {"adapters": self.adapter_statuses()}

    async def adapter_states(self, names: list[str] = Query(default_factory=list)) -> dict:
        return {"states": {name: self.backend.registry.adapter_state(name) for name in names}}

    async def get_adapter(self, name: str) -> dict:
        for status in self.adapter_statuses():
            if status["name"] == name:
                return status
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")

    async def register_adapter(self, request: RegisterAdapterRequest) -> dict:
        if (request.config is None) == (request.yaml_path is None):
            raise HTTPException(status_code=400, detail="Exactly one of 'config' or 'yaml_path' must be set")
        if request.yaml_path is not None:
            config = parse_adapter_yaml(Path(request.yaml_path))
        else:
            config = request.config
        return await self.backend.register(request.name, config)

    async def deregister_adapter(self, name: str) -> dict:
        state = self.backend.registry.adapter_state(name)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Adapter '{name}' not registered")
        await self.backend.deregister(name)
        return {"status": "ok", "name": name}
