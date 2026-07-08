"""Fast tests for AdapterRegistry + MultiLoRABackend gating/validation
(no Ray, no HTTP I/O, no SGLang, no torch)."""

import pytest

from miles.utils.multi_lora import AdapterRegistry, MultiLoRABackend, make_rid, parse_adapter


def make_backend(max_adapters: int = 4) -> MultiLoRABackend:
    return MultiLoRABackend(max_adapters, "http://unused")


def test_rid_roundtrip_preserves_names_with_underscores():
    for name in ["a", "adapter_a", "weird__name", "x_y_z"]:
        assert parse_adapter(make_rid(name)) == name


def test_register_assigns_slot_and_active_adapters_view():
    registry = AdapterRegistry(max_adapters=4)
    result = registry.register("A", config={"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}
    assert registry.active() == {"A": 0}
    view = registry.active_adapters()["A"]
    assert view.name == "A"
    assert view.slot == 0
    assert view.config == {"rm_type": "x"}


def test_forward_active_then_response_kept():
    backend = make_backend()
    backend.registry.register("A", None)
    rid = make_rid("A")
    assert backend.on_forward(rid) is True
    assert backend.on_response(rid) is False  # keep, not dummy


def test_forward_blocked_for_unknown_adapter():
    backend = make_backend()
    assert backend.on_forward(make_rid("A")) is False


def test_deregister_mid_flight_dummies_response():
    backend = make_backend()
    backend.registry.register("A", None)
    rid = make_rid("A")
    assert backend.on_forward(rid) is True
    backend.registry.deregister("A")
    assert backend.on_response(rid) is True  # dummy


def test_deregister_then_new_request_blocked():
    backend = make_backend()
    backend.registry.register("A", None)
    backend.registry.deregister("A")
    assert backend.on_forward(make_rid("A")) is False


def test_deregister_holds_slot_until_free_slot():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)
    registry.register("B", None)
    registry.deregister("A")
    assert not registry.free_slots  # slot 0 held until cleanup
    registry.free_slot("A")
    assert registry.register("C", None) == {"name": "C", "slot": 0}
    assert registry.active() == {"B": 1, "C": 0}


def test_swap_a_to_b_independent():
    backend = make_backend()
    backend.registry.register("A", None)
    rid_a = make_rid("A")
    assert backend.on_forward(rid_a) is True
    backend.registry.deregister("A")
    backend.registry.register("B", None)
    rid_b = make_rid("B")
    assert backend.on_forward(rid_b) is True
    assert backend.on_response(rid_a) is True  # straggler A dummied
    assert backend.on_response(rid_b) is False


def test_weight_version_is_globally_monotonic():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)
    assert registry.increment_weight_version() == 1
    registry.register("B", None)  # registered mid-run: stamped with current version
    assert registry.active_adapters()["A"].version == 1
    assert registry.active_adapters()["B"].version == 1
    assert registry.increment_weight_version() == 2
    registry.deregister("A")
    registry.free_slot("A")
    registry.register("A2", None)
    assert registry.increment_weight_version() == 3  # never resets on slot reuse
    assert registry.active_adapters()["A2"].version == 3


def test_step_counts_per_adapter():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)
    registry.register("B", None)
    registry.increment_steps(["A", "B"])
    registry.increment_steps(["A"])
    registry.increment_steps(["gone"])  # inactive names ignored
    assert registry.active_adapters()["A"].step == 2
    assert registry.active_adapters()["B"].step == 1
    assert registry.step_count("A") == 2

    registry.deregister("A")
    assert registry.step_count("A") == 2  # survives until free_slot (final ckpt tag)
    registry.free_slot("A")
    assert registry.step_count("A") == 0


def test_set_step_on_resume():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)
    registry.set_step("A", 40)
    registry.increment_steps(["A"])
    assert registry.step_count("A") == 41
    registry.set_step("gone", 10)  # inactive names ignored
    assert registry.step_count("gone") == 0


@pytest.mark.asyncio
async def test_custom_backend_validation_rejects():
    class StrictBackend(MultiLoRABackend):
        async def validate_adapter(self, name, config):
            if not config:
                raise ValueError("adapter config is required")

    backend = StrictBackend(4, "http://unused")
    with pytest.raises(ValueError, match="config is required"):
        await backend.register("A", None)
    assert backend.registry.active() == {}

    result = await backend.register("A", {"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}
