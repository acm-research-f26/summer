# IMPLEMENTATION 1 (LITE) — Env Setup + Ultralight Compute Check

Copy everything below this line into Claude Code as your instruction. This replaces the original Implementation 1 if `run_webshop_1gpu.sh` was still too heavy. Same installation steps as before — the only thing that changes is Step 5, which now targets a much smaller footprint and adds an inference-only checkpoint before attempting any training at all, so you find out where things break before paying for a full training attempt.

---

## Steps 1–4 (installation) — unchanged from Implementation 1
```bash
git clone https://github.com/JonP07/HiPER-agent.git
cd HiPER-agent

conda create -n verl python==3.12 -y
conda activate verl
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5
pip3 install peft==0.17.1

conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
cd agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
# verify:
python run_web_agent_text_env.py

conda activate verl
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

## Step 4.5 — make this our own project, not an upstream clone
Before doing any more work, detach this from `JonP07/HiPER-agent` as an upstream dependency and turn it into our own local project — we need to freely alter, extend, and commit to this code as our own codebase (adding the orchestrator/multi-agent layer in later implementations), not keep treating it as someone else's repo we're patching.

```bash
cd HiPER-agent
rm -rf .git
git init
git add -A
git commit -m "Initial import from JonP07/HiPER-agent (arXiv:2602.16165), detached from upstream — this is now our own project codebase"
cd ..
mv HiPER-agent acm-research-credit-rl
cd acm-research-credit-rl
```
From this point on, every implementation (this one, and Implementation 2/3 later) works inside `acm-research-credit-rl/`, not by re-cloning or pulling from the original upstream repo. If we want to push this somewhere for the team to share, create a fresh GitHub repo under our own account/org and set it as the new `origin` — do not push to or open PRs against the original `JonP07/HiPER-agent`, since this is now a divergent, independently-owned copy for our project.

**Note for later:** the licensing terms of the original repo still apply to the code we're building on (check `LICENSE` in the repo before any public writeup/release) — detaching git history doesn't change that we're building on HiPER's released code, it just means we're free to modify it as our own working copy going forward, and any writeup should still cite and credit the original HiPER paper/repo appropriately.

## Step 5 (LITE) — inference-only checkpoint before any training
Before attempting training at all, just load the model and run a handful of WebShop episodes with **no gradient updates** — pure rollout, to confirm the model, environment, and prompt structure all wire together correctly. This catches most integration bugs (prompt template issues, environment API mismatches, tokenizer/model loading problems) without spending any training compute or risking OOM from the training-specific memory overhead (optimizer states, gradients, critic).

Use `Qwen/Qwen2.5-0.5B-Instruct` for this check (smaller than the 1.5B used elsewhere — this step only needs the model to load and generate, not to be a good agent). Confirm: model loads, a WebShop episode runs to completion (success or failure, doesn't matter which), and the Plan-Execute prompt structure (`<switch>...</switch><subgoal>...</subgoal><action>...</action>`) parses correctly from the model's output.

**Report back before moving to Step 6:** did this run without error, and what GPU memory did just loading + inference use (this tells you your actual headroom before training overhead gets added).

## Step 6 (LITE) — the ultralight training attempt
I'm attaching `run_webshop_ultralight.sh` — a further-reduced config from the original 1-GPU script, changing three things to cut memory substantially:
- **Smaller model:** `Qwen2.5-0.5B-Instruct` instead of `1.5B` (roughly a 3x parameter-count reduction).
- **LoRA enabled** instead of full fine-tuning — this is a real, confirmed-supported flag in the repo's own config (`actor_rollout_ref.model.lora_rank=32`, `lora_alpha=16`, `target_modules=all-linear`, found directly in `verl/trainer/config/ppo_trainer.yaml`), not a guess. LoRA trains a small set of adapter weights instead of the full model, which cuts optimizer-state memory dramatically — this is usually the single biggest lever for fitting training on constrained hardware, bigger than batch size tweaks.
- **Smaller batches and sequence lengths:** `train_data_size=8` (2 groups of 4, down from 16), `ppo_mini_batch_size=8`, `ppo_micro_batch_size_per_gpu=2`, `max_prompt_length=1024` (down from 2048), `max_response_length=256` (down from 512).

Diff `run_webshop_ultralight.sh` against `run_webshop_1gpu.sh` yourself to see exactly what changed — it's a minimal, traceable diff, same as the earlier scripts.

Attempt to launch it. **If it still OOMs**, in this order:
1. Reduce `lora_rank` further (e.g., `16` or `8`) — lower rank means fewer trainable parameters.
2. Reduce `ppo_micro_batch_size_per_gpu` to `1`.
3. Reduce `max_prompt_length`/`max_response_length` further (WebShop pages can be long; if this becomes a real bottleneck, note it — it may mean WebShop's default page-text length needs truncating at the environment level too, not just the model's input cap).
4. Only as a last resort, reduce `train_data_size` to `4` (2 groups of 2) — this is getting close to too little signal for the group-relative advantage to mean much, so treat this as "we're now finding the actual floor," not just another dial to turn.

**Keep the `keep_penalty`/`keep_consistency_penalty` terms and the WebShop-specific `env.max_steps=50`/`bootstrap_truncated=False` settings intact throughout** — don't strip these to save memory, they don't cost meaningful compute and removing them risks the degenerate-behavior collapse they're there to prevent.

## Deliverable — report back before proceeding to Implementation 2
- Whether Step 5 (inference-only) succeeded, and what memory that alone used.
- The exact final working training config from Step 6 (every flag changed from `run_webshop_ultralight.sh`, including which fallback step you had to go to).
- Wall-clock per training iteration at that setting.
- Peak GPU memory at that setting.
- Whether reward shows any upward trend over the first ~50-100 iterations.
- **An honest assessment of whether 0.5B + LoRA is going to be viable for the rest of the project**, or whether this reveals we need to rethink something more fundamental (fewer parallel roles, a different model family, cloud burst compute for just the training runs, etc.) — flag this now rather than after Implementation 2 is built on top of it.

---

*(End of Implementation 1 LITE prompt. Attach `run_webshop_ultralight.sh`.)*
