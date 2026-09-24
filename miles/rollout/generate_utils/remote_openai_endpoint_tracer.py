"""Remote session client; requires only httpx, numpy, and safetensors."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
import safetensors.numpy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteSample:
    tokens: list[int]
    response: str
    response_length: int
    loss_mask: list[int] | None
    rollout_log_probs: list[float] | None
    status: str
    reward: float | dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteSamplesReply:
    samples: list[RemoteSample]
    session_metadata: dict[str, Any]
    empty_reason: str | None


def decode_remote_samples(payload: bytes) -> RemoteSamplesReply:
    """Read the existing v1/v2 samples response without importing Miles' trainer types."""
    tensors = safetensors.numpy.load(payload)
    metadata = tensors.pop("_samples_meta")
    if metadata.ndim != 1 or metadata.dtype != np.uint8:
        raise ValueError("invalid samples metadata: expected a one-dimensional uint8 array")
    reply = json.loads(metadata.tobytes())
    samples = []
    for index, row in enumerate(reply["samples"]):
        values = {}
        for name, dtype in (("tokens", np.int64), ("loss_mask", np.uint8), ("rollout_log_probs", np.float64)):
            if name in row["nulls"]:
                values[name] = [] if name == "tokens" else None
                continue
            array = tensors[f"{name}.{index}"]
            if array.ndim != 1 or array.dtype != dtype:
                raise ValueError(f"invalid {name} for sample {index}: shape={array.shape}, dtype={array.dtype}")
            values[name] = array.tolist()
        samples.append(
            RemoteSample(
                **values,
                response=row["response"],
                response_length=row["response_length"],
                status=row["status"],
                reward=row.get("reward"),
                metadata=row["metadata"],
            )
        )
    return RemoteSamplesReply(
        samples=samples, session_metadata=reply["session_metadata"], empty_reason=reply["empty_reason"]
    )


class RemoteOpenAIEndpointTracer:
    """Client-side equivalent of OpenAIEndpointTracer over the session HTTP API."""

    def __init__(
        self,
        session_server_url: str,
        session_id: str,
        *,
        api_key: str | None = None,
        timeout: float = 120,
        client: httpx.AsyncClient | None = None,
    ):
        self.router_url = session_server_url.rstrip("/")
        self.session_id = session_id
        self.base_url = f"{self.router_url}/sessions/{quote(session_id, safe='')}"
        self.openai_base_url = f"{self.base_url}/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout
        self.owns_client = client is None
        self.client = client if client is not None else httpx.AsyncClient()
        self.closed = False

    @property
    def session_server_id(self) -> str:
        return self.router_url.removeprefix("http://").removeprefix("https://")

    @classmethod
    async def create(
        cls,
        session_server_url: str,
        *,
        model_path: str | None = None,
        api_key: str | None = None,
        evaluation: bool = False,
        sampling_params: dict | None = None,
        timeout: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> "RemoteOpenAIEndpointTracer":
        """Allocate one session; model_path selects a saved Tinker sampler version."""
        params = {
            key: value for key, value in (sampling_params or {}).items() if key in ("temperature", "top_p", "top_k")
        }
        body = {"evaluation": evaluation, **params}
        if model_path is not None:
            body["model_path"] = model_path
        owns_client = client is None
        http_client = client if client is not None else httpx.AsyncClient()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = await http_client.post(
                f"{session_server_url.rstrip('/')}/sessions", json=body, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            tracer = cls(
                session_server_url,
                response.json()["session_id"],
                api_key=api_key,
                timeout=timeout,
                client=http_client,
            )
            tracer.owns_client = owns_client
            return tracer
        except BaseException:
            if owns_client:
                await http_client.aclose()
            raise

    async def get_session(self) -> dict:
        response = await self.client.get(self.base_url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    async def collect_samples(
        self, *, max_seq_len: int | None = None, agent_metadata: dict | None = None
    ) -> RemoteSamplesReply:
        """Collect server-assembled samples, then delete the session like the local tracer."""
        body: dict[str, Any] = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        try:
            response = await self.client.post(
                f"{self.base_url}/samples", json=body, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            return await asyncio.to_thread(decode_remote_samples, response.content)
        finally:
            try:
                await self.close()
            except Exception:
                logger.warning("Failed to delete session %s after collecting samples", self.session_id, exc_info=True)

    async def close(self) -> None:
        if self.closed:
            return
        try:
            response = await self.client.delete(self.base_url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 404:
                response.raise_for_status()
        finally:
            self.closed = True
            if self.owns_client:
                await self.client.aclose()

    async def __aenter__(self) -> "RemoteOpenAIEndpointTracer":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            await self.close()
        else:
            try:
                await self.close()
            except Exception:
                logger.warning("Failed to delete session %s", self.session_id, exc_info=True)
