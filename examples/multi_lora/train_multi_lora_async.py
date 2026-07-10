"""Custom fully-async multi-LoRA trainer.

Drives the controller (Ray actor + control-plane HTTP API) with no drain and no
rollout-id tracking:
  - create the controller (registry + API; generation traffic goes straight to
    the router),
  - register adapters from CLI (parse YAML -> AdapterConfig) + load into Megatron
    slots via ``actor_model.load_adapters``,
  - loop: read the controller snapshot, reconcile (retire + clean up),
    collect a batch from the continuous rollout, train, upsert via ``update_weights``,
  - the data source deregisters adapters at num_row; the trainer cleans them up
    (save ckpt + clear slot + free) on reconcile.

Compile-checked only; needs torch + Ray + SGLang to run.
"""

import asyncio
import logging
from pathlib import Path

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.adapter_config import parse_adapter_yaml
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils.tracking import init_tracking

from miles.ray.multi_lora_controller import create_controller, get_multi_lora_controller

logger = logging.getLogger(__name__)

ROLLOUT_FUNCTION_PATH = "examples.multi_lora.multi_lora_async_rollout.generate_rollout_multi_lora"
DATA_SOURCE_PATH = "examples.multi_lora.multi_lora_data_source_async.MultiLoRAAsyncDataSource"


async def main(args):
    assert not args.colocate, "Colocation is not supported for fully-async training (generation needs continuous GPU; colocate time-shares)."
    configure_logger(args)

    args.rollout_function_path = ROLLOUT_FUNCTION_PATH
    args.data_source_path = DATA_SOURCE_PATH
    args.rollout_global_dataset = True

    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, _num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    router_ip, router_port = await rollout_manager.get_router_address.remote()
    args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
    controller = create_controller(args, f"http://{router_ip}:{router_port}")
    await controller.start.remote()
    host = await controller.http_host.remote()
    api_port = await controller.api_port.remote()
    logger.info(f"Multi-LoRA control API listening on http://{host}:{api_port} (head node)")

    actor_model, _ = await create_training_models(args, pgs, rollout_manager)

    # Register adapters from CLI. The loop's first reconcile_adapters loads them
    # into Megatron slots and the first update_weights upserts their initial
    # weights, so the first rollout uses the adapter weights. In service mode the
    # trainer idle-waits for registrations; with --multi-lora-disable-service-mode
    # it exits when no adapters are active.
    for name, path in args.multi_lora_adapters:
        config = parse_adapter_yaml(Path(path))
        await controller.register_adapter.remote(name, config)

    rollout_id = 0
    while True:
        snapshot = await get_multi_lora_controller().snapshot.remote()
        if not (snapshot["pending"] or snapshot["active"] or snapshot["retiring"] or snapshot["cleanup"]):
            if not args.multi_lora_service_mode:
                logger.info("No adapters; exiting.")
                break
            logger.info(f"No adapters; sleeping for {args.multi_lora_idle_poll_s}s...")
            await asyncio.sleep(args.multi_lora_idle_poll_s)
            continue

        # Reconcile (load active+pending, cleanup deregistered) then push
        # BEFORE generate. The weight updater reports what it pushed, which
        # promotes pending adapters to active — only then does the data
        # source start sampling them.
        await actor_model.reconcile_adapters()
        await actor_model.update_weights()

        # Reconcile may have cleaned up the last adapters (e.g. num_row reached):
        # with nothing active, generate would wait forever for a batch.
        post_update = await get_multi_lora_controller().snapshot.remote()
        if not (post_update["active"] or post_update["retiring"]):
            continue

        rollout_data = await rollout_manager.generate.remote(rollout_id)
        await actor_model.train(rollout_id, rollout_data)

        # Save cadence is per adapter, decided inside save_model from each
        # adapter's own step count; deregistration cleanup saves final ckpts.
        await actor_model.save_model(rollout_id)

        rollout_id += 1

    await rollout_manager.dispose.remote()
    await controller.stop.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
