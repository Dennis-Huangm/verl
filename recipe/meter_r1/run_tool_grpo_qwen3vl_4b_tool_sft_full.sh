#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3vl_4b_tool_sft_tool_grpo_full_v2}
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/verl_checkpoints/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${REPO_ROOT}/outputs/verl_rollout_logs/${EXPERIMENT_NAME}"
LOG_DIR="${REPO_ROOT}/logs/verl/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}/verl"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${REPO_ROOT}/outputs/verl_parquet/final_sft_v6_rl5400_tool/train.parquet" \
    data.val_files="${REPO_ROOT}/outputs/verl_parquet/final_sft_v6_rl5400_tool/test.parquet" \
    data.image_key=images \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path="${REPO_ROOT}/checkpoints/final_sft_v6_train_combined_b11_recovery_gpt_v3_skeletonprompt_20260529_qwen3vl4b_full_e2_bs4" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=14336 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/meter_r1/crop_image_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="${REPO_ROOT}/data_synthesis/tool_grpo_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=meter_r1_tool_grpo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.log_val_generations=8 \
    +trainer.log_train_generations=8 \
    +trainer.log_train_freq=20 \
    +ray_kwargs.ray_init.address=local \
    "$@"
