"""Multi-LoRA Ray actor + named-actor lookup.

The controller logic + HTTP server live in ``miles.multi_lora`` (no Ray). This
module wraps them in a named Ray actor (so library code reaches it via
``get_multi_lora_controller()``) and runs the HTTP server out-of-band.
"""

import time
from functools import cache
from typing import Any

import ray

from miles.utils.multi_lora import MultiLoRAControllerLogic, MultiLoRAHTTPServer

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


@ray.remote(num_cpus=0)
class MultiLoRAAsyncController:
    def __init__(self, args, upstream_url: str, host: str = "0.0.0.0", port: int = 0) -> None:
        self.logic = MultiLoRAControllerLogic(args.multi_lora_n_adapters)
        self.server = MultiLoRAHTTPServer(self.logic, upstream_url, host, port)

    async def start(self) -> int:
        await self.server.start()
        return self.server.actual_port

    async def stop(self) -> None:
        await self.server.stop()

    def register_adapter(self, name: str, config: Any) -> dict:
        return self.logic.register_adapter(name, config)

    async def deregister_adapter(self, name: str) -> None:
        self.logic.deregister_adapter(name)
        await self.server.abort_adapter_requests(name)

    def free_slot(self, name: str) -> int:
        return self.logic.free_slot(name)

    def increment_slot_version(self, name: str) -> None:
        self.logic.increment_slot_version(name)

    def active_adapters(self) -> dict:
        return self.logic.active_adapters()

    def active(self) -> dict:
        return self.logic.active()

    def http_host(self) -> str:
        return self.server.host

    def http_port(self) -> int:
        return self.server.actual_port


def create_controller(args, upstream_url: str, host: str = "0.0.0.0", port: int = 0):
    return MultiLoRAAsyncController.options(
        name=CONTROLLER_NAME, namespace=CONTROLLER_NAMESPACE
    ).remote(args, upstream_url, host, port)
