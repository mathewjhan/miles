"""Fast tests for AdapterRegistry + MultiLoRABackend gating/validation (no Ray,
no HTTP I/O, no SGLang, no torch). The backend constructor opens no sockets, so
everything except the engine-facing abort is testable without starting it."""

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
    assert backend.on_response(rid) is False  # A still active -> keep


def test_forward_blocked_for_unknown_adapter():
    backend = make_backend()
    assert backend.on_forward(make_rid("A")) is False  # never registered -> block


def test_deregister_mid_flight_dummies_response():
    backend = make_backend()
    backend.registry.register("A", None)
    rid = make_rid("A")
    assert backend.on_forward(rid) is True
    backend.registry.deregister("A")  # removed mid-flight
    assert backend.on_response(rid) is True  # A gone -> dummy


def test_deregister_then_new_request_blocked():
    backend = make_backend()
    backend.registry.register("A", None)
    backend.registry.deregister("A")
    assert backend.on_forward(make_rid("A")) is False


def test_deregister_holds_slot_until_free_slot():
    registry = AdapterRegistry(max_adapters=2)
    registry.register("A", None)  # slot 0
    registry.register("B", None)  # slot 1
    registry.deregister("A")  # slot 0 held, not freed
    assert not registry.free_slots  # no free slots (0 held, 1 in use)
    registry.free_slot("A")  # trainer cleanup -> slot 0 freed
    assert registry.register("C", None) == {"name": "C", "slot": 0}  # reuses freed slot
    assert registry.active() == {"B": 1, "C": 0}


def test_swap_a_to_b_independent():
    backend = make_backend()
    backend.registry.register("A", None)
    rid_a = make_rid("A")
    assert backend.on_forward(rid_a) is True
    backend.registry.deregister("A")
    backend.registry.register("B", None)  # reuses slot 0
    rid_b = make_rid("B")
    assert backend.on_forward(rid_b) is True  # B active -> forward
    assert backend.on_response(rid_a) is True  # straggler A -> dummy
    assert backend.on_response(rid_b) is False  # B -> keep


@pytest.mark.asyncio
async def test_custom_backend_validation_rejects():
    class StrictBackend(MultiLoRABackend):
        async def validate_adapter(self, name, config):
            if not config:
                raise ValueError("adapter config is required")

    backend = StrictBackend(4, "http://unused")
    with pytest.raises(ValueError, match="config is required"):
        await backend.register("A", None)
    assert backend.registry.active() == {}  # rejected before touching the registry

    result = await backend.register("A", {"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}
