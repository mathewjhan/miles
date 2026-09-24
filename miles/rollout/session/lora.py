"""Resolve saved Tinker sampler weights and load them on the inference workers."""

import asyncio
from dataclasses import dataclass

import httpx

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError, UpstreamResponseError
from miles.tinker.core.types import UserInputError
from miles.tinker.core.utils import resolve_sampler_checkpoint
from miles.utils.lora.utils import lora_rollout_enabled


@dataclass(frozen=True)
class SessionLoRA:
    name: str
    path: str


def request_api_key(headers) -> str:
    return headers.get("x-api-key") or (headers.get("authorization") or "").removeprefix("Bearer ").strip()


def resolve_session_lora(config: SessionServerConfig, *, model_path: str, api_key: str) -> SessionLoRA:
    if not lora_rollout_enabled(config):
        raise MessageValidationError("session adapters require LoRA-enabled inference engines")
    if not config.tinker_checkpoint_root:
        raise MessageValidationError("model_path requires a configured Tinker checkpoint root")
    if not api_key:
        raise MessageValidationError("model_path requires the Tinker API key in X-API-Key or Authorization")
    try:
        name, path = resolve_sampler_checkpoint(
            config.tinker_checkpoint_root, api_key, model_path, config.tinker_base_model or config.hf_checkpoint
        )
    except (UserInputError, AssertionError) as error:
        raise MessageValidationError(str(error)) from error
    return SessionLoRA(name=name, path=path)


async def load_session_lora(client: httpx.AsyncClient, backend_url: str, lora: SessionLoRA) -> None:
    """Preload all current workers; SGLang can subsequently reload evicted adapters from its saved references."""
    try:
        response = await client.get(f"{backend_url}/workers")
        if response.status_code == 404:
            response = await client.get(f"{backend_url}/list_workers")
            if response.status_code == 404:
                urls = [backend_url]
            else:
                response.raise_for_status()
                urls = response.json()["urls"]
        else:
            response.raise_for_status()
            urls = [worker["url"] for worker in response.json()["workers"]]
        if not urls:
            raise UpstreamResponseError("no inference workers are available to load the session adapter")
        responses = await asyncio.gather(
            *[
                client.post(
                    f"{url.rstrip('/')}/load_lora_adapter",
                    json={"lora_name": lora.name, "lora_path": lora.path, "pinned": False},
                )
                for url in dict.fromkeys(urls)
            ]
        )
        for response in responses:
            result = response.json()
            if result.get("success") is True and response.is_success:
                continue
            # Independent sessions may share the same immutable saved version.
            if response.status_code == 400 and f"{lora.name} because it is already loaded" in str(
                result.get("error_message", "")
            ):
                continue
            raise UpstreamResponseError(f"failed to load session adapter {lora.name}: {response.text}")
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise UpstreamResponseError(f"failed to load session adapter {lora.name}: {error}") from error
