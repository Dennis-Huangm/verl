#!/usr/bin/env bash
# 2GPU full-parameter no-LoRA full-offload config; this still OOMs during actor optimizer step.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/run_grpo_qwen3vl_8b_final_sft_smoke.sh" \
    trainer.experiment_name=qwen3vl_8b_base_pure_rl_smoke_fullparam_2gpu_no_lora_ctx2048_full_offload_oom \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    "$@"
