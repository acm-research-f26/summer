set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# PAPER-SCALE REPLICATION ATTEMPT, not a smoke test -- goal is to reproduce
# HiPER's own reported 95.3% ALFWorld success rate at Qwen2.5-1.5B
# (implementation-1/UPSTREAM_README.md's Results Summary table), using
# Paper Baseline (this vendored verl/Ray/FSDP/vLLM trainer, critic-based HAE,
# single-agent Plan-Execute -- NOT the orchestrator/delegation idea, which
# this paper never tested; see PROJECT_GUIDE.md section 9).
#
# Base: run_alfworld.sh (implementation-1's untouched copy of HiPER's own
# reference script -- the one that actually produced 95.3%), kept as close
# to verbatim as this hardware allows. Every value below is either identical
# to that reference or has an explicit comment explaining why it had to
# change. This is deliberately a *different* script from run_alfworld_lite.sh
# (which stays as the fast structural-correctness smoke config) rather than
# an edit to it -- the lite config's whole purpose was fast iteration at a
# scale with no published number to compare against; this one's purpose is
# the opposite.
#
# THE FORCED DEVIATIONS: this machine has 2 GPUs (2x RTX 6000 Ada, 49GB
# each), not the reference's 4. trainer.n_gpus_per_node is dropped 4 -> 2
# accordingly; tensor_model_parallel_size=2 is kept (vLLM rollout uses both
# GPUs as one TP group, same as the reference's per-replica TP degree), which
# means -- unlike the reference's 4-GPU setup, which ran 2 rollout replicas
# in parallel -- this runs a single rollout replica. Expect roughly on the
# order of 2x the reference's wall-clock per epoch from reduced parallelism
# alone, on top of whatever the actual per-step compute time turns out to be.
#
# A first launch attempt with the reference's own actor/critic memory settings
# (fsdp_config.param_offload=False, optimizer_offload=False -- fine on their 4
# GPUs) OOM'd during the actor's loss.backward() on GPU 0 (47.37GB total, 407MB
# free at the crash). Fixed by turning FSDP param/optimizer offload ON for both
# actor and critic (same fallback run_alfworld_lite.sh already uses at 0.5B)
# and halving ppo_micro_batch_size_per_gpu / critic.ppo_micro_batch_size_per_gpu
# 16 -> 8 to shrink backward-pass activation peak memory. These trade some
# speed for fitting in half the reference's GPU count -- a real, necessary
# deviation, not a downtune of what's being measured (mini-batch size, lr, KL
# coefficients, and every HAE/reward hyperparameter stay exactly the reference's
# values).
#
# TWO DELIBERATE OPERATIONAL ADDITIONS (not in the reference, which was
# presumably run interactively/monitored): trainer.save_freq is set to 10
# (reference uses -1, never checkpoint) and trainer.resume_mode='auto', so
# an unattended multi-hour-to-multi-day run started via a background process
# can be restarted from its last checkpoint rather than losing all progress.
# trainer.logger stays ['console'] only (not ['console','wandb']) -- wandb.init()
# needs an interactive login not available in this environment, same
# constraint run_alfworld_lite.sh already documented.

num_cpus_per_env_worker=0.1

train_data_size=128 # match GRPO/GiGPO configuration (16 x 8), identical to the reference
val_data_size=128

# run_alfworld.sh's own module path (`examples.data_preprocess.prepare`) doesn't
# exist in this repo (no `examples/` dir, a leftover from the original upstream
# layout) -- same real path bug already fixed in run_alfworld_lite.sh, fixed
# here identically (unrelated to model/GPU scale).
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
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.use_two_heads_critic=False \
    critic.use_three_heads_critic=True \
    algorithm.use_kl_in_reward=False \
    algorithm.hae.norm_adv=True \
    algorithm.hae.keep_penalty=-0.3 \
    algorithm.hae.keep_consistency_penalty=-0.3 \
    algorithm.hae.bootstrap_truncated=True \
    env.env_name=alfworld/AlfredTWEnvOptions \
    env.seed=6 \
    env.max_steps=50 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    reward_model.reward_manager=multi_turn \
    trainer.logger=['console'] \
    trainer.log_val_generations=10 \
    trainer.project_name='hiper_alfworld_paper_scale' \
    trainer.experiment_name='hiper_qwen2.5_1.5b_full_ft_2gpu' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.resume_mode='auto' \
    trainer.val_before_train=True $@
