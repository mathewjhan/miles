"""Tests for the testable core of the multi-LoRA async rollout (process_group).

rid stamping lives in ``generate`` (next to ``lora_path``), not here, so these
tests cover keep-vs-recycle plus slot-version stamping. The slot version must be
captured at *submission* (before generation): in fully-async another update can
bump the controller version during generation, and re-querying afterwards would
make a stale group look fresh to the staleness filter.
"""

import pytest

from miles.utils.types import AdapterRef, Sample

import examples.multi_lora.multi_lora_async_rollout as mod
from examples.multi_lora.multi_lora_async_rollout import process_group


class FakeDataSource:
    def __init__(self) -> None:
        self.added: list = []

    def add_samples(self, groups) -> None:
        self.added.extend(groups)


class FakeVersionCache:
    def __init__(self, versions: dict[str, int]) -> None:
        self.versions = versions

    def bump(self, name: str, to: int) -> None:
        self.versions[name] = to

    async def get_all(self) -> dict[str, int]:
        return dict(self.versions)

    async def get(self, adapter_name: str) -> int | None:
        return self.versions.get(adapter_name)


def group(adapter: str = "A", slot: int = 0) -> list[Sample]:
    return [Sample(prompt="p", adapter=AdapterRef(adapter, slot))]


@pytest.mark.asyncio
async def test_process_group_keeps_completed():
    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.COMPLETED
        return group

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is g
    assert ds.added == []


@pytest.mark.asyncio
async def test_process_group_recycles_aborted():
    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.ABORTED
        return group

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is None
    assert len(ds.added) == 1  # recycled back to the data source


@pytest.mark.asyncio
async def test_process_group_stamps_submission_version(monkeypatch):
    """The stamp is the version live at submission (5), not completion (7)."""
    cache = FakeVersionCache({"A": 5})

    async def gen(args, group, sampling_params):
        # Another train/update cycle fires mid-generation, bumping the version.
        cache.bump("A", 7)
        for s in group:
            s.status = Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(mod, "slot_version_cache", cache)

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is g
    assert g[0].metadata["slot_version"] == 5  # submission version, not 7


@pytest.mark.asyncio
async def test_process_group_no_adapter_skips_stamp(monkeypatch):
    """A group with no adapter binding must not query the version cache or
    write a slot_version."""

    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.COMPLETED
        return group

    class FailingCache:
        async def get(self, adapter_name):
            raise AssertionError("version cache should not be queried for adapter-less group")

    monkeypatch.setattr(mod, "slot_version_cache", FailingCache())

    ds = FakeDataSource()
    g = [Sample(prompt="p", adapter=None)]
    result = await process_group(None, g, {}, gen, ds)

    assert result is g
    assert "slot_version" not in g[0].metadata
