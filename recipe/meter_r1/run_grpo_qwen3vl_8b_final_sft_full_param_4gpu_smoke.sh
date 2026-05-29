#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/run_grpo_qwen3vl_8b_final_sft_smoke.sh" \
    trainer.experiment_name=qwen3vl_8b_base_pure_rl_smoke_fullparam_4gpu_no_lora_ctx2048 \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    "$@"
