#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/run_grpo_qwen3vl_8b_final_sft_smoke.sh" \
    trainer.experiment_name=qwen3vl_8b_base_pure_rl_smoke_lora_merge_2gpu_rank64 \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    "$@"
