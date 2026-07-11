# Implementation 1 (LITE) — Environment Setup + Ultralight Compute Check


## What this phase was for

Before attempting any training, confirm the whole pipeline — model, WebShop environment, Plan-Execute prompt format, and the training loop itself — actually runs end-to-end at a compute scale this project can afford (a single GPU, 0.5B model, LoRA). This phase deliberately does **not** aim for tuned or converged results; it exists to catch integration problems early and establish an honest read on what's actually feasible, before Implementation 2 builds the full framework on top of it.

## What we did

1. **Cloned and detached** HiPER-agent into this project as a plain subfolder (Apache 2.0 licensed — see `UPSTREAM_README.md` in this folder for the original project's citation), not a re-clonable upstream dependency.
2. **Installed the environment.** WebShop's Ray actor runs *in-process* inside the trainer rather than in a separate process/env — this meant installing WebShop's full runtime stack (gym, beautifulsoup4, spaCy + models, pyserini, faiss-cpu, OpenJDK) into the same conda environment as the training stack (torch/flash-attn/vLLM/peft), not two separate environments as originally assumed.
3. **Built the WebShop working dataset**: a 1,000-product subset drawn from the full ~1.18M-product WebShop corpus, since the pre-built subset files were blocked by Google Drive's download quota.
4. **Step 5 — inference-only check**: confirmed Qwen2.5-0.5B-Instruct loads (~1GB standalone) and a full WebShop episode runs to completion through the real Plan-Execute prompt pipeline, with zero gradient updates.
5. **Step 6 — ultralight training attempt**: launched the reduced-memory training config (0.5B model, LoRA rank 32, small batch/group sizes) and completed 2 real PPO training iterations before stopping deliberately (see Results below).

## How to reproduce

```bash
conda activate verl   # single env with both the training stack and WebShop's runtime deps, see ../implementation-2/README.md
cd ../implementation-2

# Step 5 -- inference-only check
python agent_system/environments/env_package/webshop/webshop/run_web_agent_text_env.py

# Step 6 -- ultralight training attempt (~4 min/iteration)
bash example_scripts/HiPER_trainer/run_webshop_ultralight.sh
```

