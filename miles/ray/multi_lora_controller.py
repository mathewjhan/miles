"""Multi-LoRA Ray actor + named-actor lookup.

The adapter registry + HTTP server live in ``miles.utils.multi_lora`` (no Ray). This
module wraps them in a named Ray actor (so library code reaches it via
``get_multi_lora_controller()``) and runs the HTTP server out-of-band.
"""

import time
from functools import cache
from typing import Any

import ray

from miles.utils.misc import load_function
from miles.utils.multi_lora import MultiLoRABackend, MultiLoRAHTTPServer

CONTROLLER_NAME = "miles_multi_lora_controller"
CONTROLLER_NAMESPACE = "miles"


@cache
def get_multi_lora_controller():
    return ray.get_actor(CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE)


class SlotVersionCache:
    """TTL-cached snapshot of the controller's active adapters -> slot version."""

    def __init__(self, ttl_s: float = 1.0) -> None:
        self.ttl_s = ttl_s
        self.versions: dict[str, int] = {}
        self.last_refresh: float | None = None

    async def get_all(self) -> dict[str, int]:
        now = time.monotonic()
        if self.last_refresh is None or now - self.last_refresh >= self.ttl_s:
            try:
                adapters = await get_multi_lora_controller().active_adapters.remote()
                self.versions = {name: adapter.version for name, adapter in adapters.items()}
                self.last_refresh = now
            except Exception:
                pass
        return self.versions

    async def get(self, adapter_name: str) -> int | None:
        return (await self.get_all()).get(adapter_name)


slot_version_cache = SlotVersionCache()


def _load_subclass(path: str | None, base_cls):
    if not path:
        return base_cls
    cls = load_function(path)
    assert issubclass(cls, base_cls), f"{path} must point to a {base_cls.__name__} subclass, got {cls}"
    return cls


@ray.remote(num_cpus=0)
class MultiLoRAController:
    def __init__(self, args, upstream_url: str, host: str = "0.0.0.0", port: int = 0) -> None:
        backend_cls = _load_subclass(getattr(args, "multi_lora_backend_path", None), MultiLoRABackend)
        server_cls = _load_subclass(getattr(args, "multi_lora_http_server_path", None), MultiLoRAHTTPServer)
        self.backend = backend_cls(args.multi_lora_n_adapters, upstream_url)
        self.server = server_cls(
            self.backend, host, port, api_port=getattr(args, "multi_lora_api_port", 0)
        )

    async def start(self) -> int:
        await self.backend.init()
        await self.server.start()
        return self.server.actual_port

    async def stop(self) -> None:
        await self.server.stop()
        await self.backend.close()

    async def register_adapter(self, name: str, config: Any) -> dict:
        return await self.backend.register(name, config)

    async def deregister_adapter(self, name: str) -> None:
        await self.backend.deregister(name)

    def free_slot(self, name: str) -> int:
        return self.backend.registry.free_slot(name)

    def increment_slot_version(self, name: str) -> None:
        self.backend.registry.increment_version(name)

    def active_adapters(self) -> dict:
        return self.backend.registry.active_adapters()

    def active(self) -> dict:
        return self.backend.registry.active()

    def http_host(self) -> str:
        return self.server.host

    def http_port(self) -> int:
        return self.server.actual_port

    def api_port(self) -> int:
        return self.server.actual_api_port


def create_controller(args, upstream_url: str, host: str = "0.0.0.0", port: int = 0):
    return MultiLoRAController.options(
        name=CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE
    ).remote(args, upstream_url, host, port)
