set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# STRUCTURAL-CORRECTNESS SMOKE CONFIG, not a convergence run -- see
# implementation-3/README.md for the reasoning. Per an explicit project
# decision (downtune GPU-heavy work for now so there's time to validate the
# ALFWorld port and the SEAL variant structurally), this deliberately does
# NOT attempt the multi-hour convergence run implementation-1's
# run_alfworld.sh reference script (150 epochs, 4 GPUs, 1.5B, no LoRA) or the
# older IMP3-tuning-phase.md plan called for. That doc's "hours, not minutes"
# framing is superseded by this decision for this session -- see the README.
#
# Base: run_alfworld.sh's env/reward/algorithm block, kept verbatim (these
# are HiPER's own correct ALFWorld reference values -- NOT copied from the
# WebShop scripts, which use different numbers for some of these, e.g.
# keep_penalty=-3.5 there vs -0.3 here).
#
# Overridden on top: model/LoRA forced to the project's verified lightweight
# scale (Qwen2.5-0.5B-Instruct + LoRA rank 32) on BOTH actor and critic --
# neither run_alfworld.sh (1.5B, no LoRA at all) nor even the WebShop
# "ultralight" script (LoRA on actor only, critic fully fine-tuned) get this
# right; fixed here rather than repeated. Two flags run_alfworld.sh leaves to
# implicit yaml defaults (keep_penalty_mode, alfworld.eval_dataset) are
# pinned explicitly below for reproducibility.
#
# trainer.logger is console-only (not ['console','wandb']) -- wandb.init()
# requires an interactive login/API key, which broke an unattended launch
# check of this exact script; not appropriate for a smoke-test config.

num_cpus_per_env_worker=0.1

train_data_size=8 # downtuned: 2 groups x 4, matching the WebShop ultralight script's own precedent
val_data_size=8

# NOTE: run_alfworld.sh's own module path (`examples.data_preprocess.prepare`)
# does not exist in this repo (no `examples/` dir) -- it's a leftover from the
# original upstream layout. Use the actual path, same fix already applied to
# the WebShop ultralight script.
python3 -m example_scripts.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# For env_name, HiPER requires alfworld/AlfredTWEnvOptions, which incorporates the Plan-Execute prompting.
# For reward_manager, we use multi_turn, which is different from the default episode reward manager (used in original verl-agent repo), to support multi-turn feedback.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=hae \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.lora_rank=32 \
    critic.model.lora_alpha=16 \
    critic.model.target_modules=all-linear \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.use_two_heads_critic=False \
    critic.use_three_heads_critic=True \
    algorithm.use_kl_in_reward=False \
    algorithm.hae.norm_adv=True \
    algorithm.hae.keep_penalty=-0.3 \
    algorithm.hae.keep_penalty_mode=normalized \
    algorithm.hae.keep_consistency_penalty=-0.3 \
    algorithm.hae.bootstrap_truncated=True \
    env.env_name=alfworld/AlfredTWEnvOptions \
    env.seed=6 \
    env.max_steps=50 \
    env.alfworld.eval_dataset=eval_in_distribution \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    reward_model.reward_manager=multi_turn \
    trainer.logger=['console'] \
    trainer.log_val_generations=10 \
    trainer.project_name='hiper_alfworld_lite' \
    trainer.experiment_name='hiper_qwen2.5_0.5b_lora_lite' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=4 \
    trainer.resume_mode='disable' \
    trainer.val_before_train=True $@
