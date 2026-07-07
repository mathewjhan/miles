# Multi-LoRA Training Example (fully-async)

Train multiple LoRA adapters concurrently against a shared base model, using a
fully-async rollout (continuous producer) + a slot-keyed LoRA page table on the
SGLang engines (in-place upsert, no unload, no drain).

This example trains two adapters on Qwen3-4B:

- **gsm8k** — grade-school math, `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), `rm_type: deepscaler`

## Layout

```
provision.sh                         # one-time: download model + datasets
single_run.sh / single_run_disagg.sh # entrypoints: bounded run, exits when done
train_multi_lora_async.py            # trainer (entry point)
controller.py                        # controller logic + HTTP proxy (torch-free, tested)
controller_actor.py                  # Ray actor wrapper (named, + HTTP out-of-band)
multi_lora_async_rollout.py          # fully-async rollout function
multi_lora_data_source_async.py      # data source (reads controller, deregisters at num_row)
test_controller_*.py                 # controller logic + HTTP tests (no torch)
adapters/
  gsm8k.yaml
  dapo_math.yaml
```

## Design (no drain, no state machine)

- **Controller** (Ray actor + HTTP proxy) is the source of truth: `register_adapter` /
  `deregister_adapter` / `active_adapters`. The data source reads it; the trainer reads
  it; rollout requests are proxied through it (blocks deregistered adapters, dummies
  in-flight stragglers via the `rid = {adapter}_{uuid}` set in `generate`).
- **No drain / no rollout-id / no train_steps / no PENDING-DRAINING-DRAINED states.**
  The data source deregisters an adapter at `num_row`; the trainer's
  `reconcile_adapters` (before each generate) cleans up gone adapters (save ckpt +
  clear Megatron slot) and loads new ones. `update_weights` upserts active adapters'
  weights in place (SGLang page table, `upsert=True`).
- **Batch ⊆ loaded property:** `reconcile_adapters` runs before `generate`, so the
  batch is fetched with loaded = active; active only shrinks during generate, so every
  adapter in the batch is live on the trainer.

## Provision (once)

```bash
bash examples/multi_lora/provision.sh
```

Downloads `Qwen/Qwen3-4B`, `zhuzilin/dapo-math-17k`, and `zhuzilin/gsm8k`.

## Run

```bash
bash examples/multi_lora/single_run.sh
```

Registers the two adapters from CLI flags and trains until each hits its `num_row`
(or `--num-rollout`), then exits.

## Multi-LoRA CLI flags

| Flag | Purpose |
| --- | --- |
| `--multi-lora-n-adapters N` | Max concurrent adapter slots. `0` disables (default); `> 0` enables. |
| `--multi-lora-adapter NAME PATH` | Register an adapter at startup. Repeatable. `PATH` → an `adapter.yaml`. |

Per-adapter `rank` in `adapter.yaml` must be `<= --lora-rank`.

## adapter.yaml

```yaml
rank: 16
alpha: 16
data: /root/gsm8k/train.parquet
input_key: messages
label_key: label
rm_type: math
num_row: 400                # stop adapter after N rows
# optional: save, num_epoch, custom_rm_path, ...
```
