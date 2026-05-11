import json
import logging
import os
from collections.abc import Mapping

import ray
import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.utils.adapter_config import AdapterConfig

logger = logging.getLogger(__name__)


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    if "lora_A" in hf_name and adapter_rank < tensor.shape[0]:
        remainder = tensor[adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_A padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:adapter_rank]
    if "lora_B" in hf_name and adapter_rank < tensor.shape[1]:
        remainder = tensor[:, adapter_rank:]
        assert remainder.abs().max() == 0, (
            f"lora_B padded dims are non-zero: {hf_name}, "
            f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
        )
        return tensor[:, :adapter_rank]
    return tensor


def save_multi_lora_checkpoints(
    args,
    model,
    iteration: int,
    adapter_configs: Mapping[str, AdapterConfig],
):
    """Save per-adapter checkpoints in two formats per adapter.

    Layout (per adapter)::

        {adapter.dir}/checkpoints/step_{iteration}/
        ├── adapter_megatron_tp{tp}_pp{pp}.pt   ← per-rank shard, fast resume
        ├── adapter_model.safetensors           ← gathered HF, inference / external
        └── adapter_config.json                 ← HF PEFT metadata (r, alpha, ...)
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot
    from megatron.core import mpu
    from safetensors.torch import save_file as save_safetensors

    from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
    from miles.utils import megatron_bridge_utils

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    is_dp_rank_0 = get_parallel_state().intra_dp.rank == 0
    is_global_writer = is_dp_rank_0 and tp_rank == 0 and pp_rank == 0

    target_modules_hf = (
        convert_target_modules_to_hf(list(args.target_modules))
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for adapter_name, config in adapter_configs.items():
        log_prefix = f"[multilora] ({adapter_name})"

        final_dir = config.dir / "checkpoints" / f"step_{iteration}"
        tmp_dir = config.dir / "checkpoints" / f"_tmp_step_{iteration}"
        if is_dp_rank_0:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        with expose_adapter_slot(model, config.slot):
            # Megatron checkpoints
            if is_dp_rank_0:
                shard: dict[str, torch.Tensor] = {
                    name: param.data.cpu()
                    for chunk in model
                    for name, param in chunk.named_parameters()
                    if ".adapter." in name
                }
                native_path = tmp_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
                torch.save(shard, native_path)
                logger.info(
                    f"{log_prefix} saved Megatron shard "
                    f"({len(shard)} tensors) to {native_path}"
                )

            hf_state: dict[str, torch.Tensor] = {}
            with megatron_bridge_utils.patch_megatron_model(model):
                for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                    model, cpu=True, show_progress=False,
                ):
                    # Safetensors format can't save aliased tensors, so need clone()
                    hf_state[hf_name] = weight.clone()

        if is_global_writer:
            save_safetensors(
                hf_state,
                str(tmp_dir / "adapter_model.safetensors"),
                metadata={"format": "pt"},
            )
            adapter_config_json = {
                "peft_type": "LORA",
                "r": config.rank,
                "lora_alpha": config.alpha,
                "target_modules": target_modules_hf,
                "lora_dropout": getattr(args, "lora_dropout", 0.0),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(tmp_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config_json, f, indent=2)
            os.sync()
            logger.info(
                f"{log_prefix} saved HF PEFT to {tmp_dir} "
                f"({len(hf_state)} tensors)"
            )

        if dist.is_initialized():
            dist.barrier()

        # to avoid partially complete checkpoints, move the checkpoint to the
        # actual directory after everything is complete
        if is_global_writer:
            if final_dir.exists():
                import shutil
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            logger.info(f"{log_prefix} promoted checkpoint to {final_dir}")
        if dist.is_initialized():
            dist.barrier()


def _register_adapter(name: str, config: AdapterConfig, model) -> None:
    """Install one PENDING adapter on this rank's local model shard.
    """
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    from ..multi_lora import find_latest_checkpoint

    log_prefix = f"[multilora] ({name})"

    ckpt_root = config.dir / "checkpoints"
    ckpt = find_latest_checkpoint(ckpt_root)
    if ckpt is None:
        logger.info(f"{log_prefix} no checkpoint under {ckpt_root}, starting from random init")
    else:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded = load_adapter(model, config.slot, state_dict)
        assert loaded > 0, (
            f"{log_prefix} loaded 0 tensors from {ckpt} "
            f"(state_dict has {len(state_dict)} entries) — name mismatch?"
        )
        logger.info(f"{log_prefix} loaded from {ckpt} ({loaded} tensors)")

    init_adapter_slot(model, config.slot, rank=config.rank, alpha=config.alpha)
    logger.info(f"{log_prefix} installed at slot {config.slot}")


def _deregister_adapter(name: str, config: AdapterConfig, rollout_id: int, args, model, optimizer) -> None:
    """Model-side cleanup for one DRAINED adapter.
    """
    from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot

    from ..multi_lora import zero_optimizer_state_for_adapter

    log_prefix = f"[multilora] ({name})"

    train_steps = ray.get(get_multi_lora_controller().adapter_train_steps.remote())
    step = train_steps[name]

    # Save the checkpoint
    save_multi_lora_checkpoints(args, model, step, {name: config})
    logger.info(f"{log_prefix} saved final checkpoint")

    # Clear out the multilora slot in the multilora layer in the Megatron model
    clear_adapter_slot(model, config.slot)
    logger.info(f"{log_prefix} cleared adapter slot {config.slot}")

    # Zero out the optimizer state to prevent future adapters from reusing previous adapter
    # momentum, etc
    zero_optimizer_state_for_adapter(optimizer, model, config.slot)
    optimizer.reload_model_params()
    logger.info(f"{log_prefix} cleared optimizer state for slot {config.slot}")


def _adapters_in_state(state):
    configs = ray.get(get_multi_lora_controller().adapter_configs.remote())
    return [(n, c) for n, c in configs.items() if c.state == state]


def load_pending_adapters(args, model, optimizer) -> int:
    from miles.backends.megatron_utils.initialize import is_megatron_main_rank
    from miles.utils.adapter_config import AdapterState
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    pending = _adapters_in_state(AdapterState.PENDING)
    if not pending:
        return 0

    for name, config in pending:
        _register_adapter(name, config, model)

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())

    if is_megatron_main_rank():
        for name, _ in pending:
            ray.get(get_multi_lora_controller().update_adapter_state.remote(name, AdapterState.ACTIVE))
    optimizer.reload_model_params()
    return len(pending)


def unload_drained_adapters(args, model, optimizer, rollout_id: int) -> int:
    """DRAINED adapters model-side cleanup.
    """
    from miles.backends.megatron_utils.initialize import is_megatron_main_rank
    from miles.utils.adapter_config import AdapterState
    from miles.utils.distributed_utils import get_gloo_group

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    drained = _adapters_in_state(AdapterState.DRAINED)
    if not drained:
        return 0
    for name, config in drained:
        _deregister_adapter(name, config, rollout_id, args, model, optimizer)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    if is_megatron_main_rank():
        for name, _ in drained:
            ray.get(get_multi_lora_controller().mark_removed.remote(name))
    return len(drained)
