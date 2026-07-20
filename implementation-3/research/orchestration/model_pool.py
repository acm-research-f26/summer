# IMP2/IMP3 -- shared model loading for the 2-role orchestration. Ported
# unchanged from implementation-2's WebShop version into implementation-3's
# ALFWorld port -- this file is environment-agnostic (no WebShop/ALFWorld
# references), only the specialists/orchestrator that select adapters here
# differ per environment.
#
# One base model (Qwen2.5-0.5B-Instruct, per Implementation 1 LITE's confirmed
# config) with TWO named LoRA adapters on top, via peft's multi-adapter support:
#   - "shared":       used by both sub-agent specialists (navigate/interact in
#                      implementation-3, search/compare in implementation-2).
#   - "orchestrator":  used only by the orchestrator.
#
# Two separate adapters (not one shared by all three roles) is what makes
# Baseline B's "sub-agents held frozen" requirement (IMP2 Part 4) actually
# possible: freezing means disabling grad on the "shared" adapter's params
# while training only "orchestrator"'s, which only works if they're distinct
# adapters. All three roles still share the same frozen base weights.

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # confirmed by Implementation 1 LITE
LORA_RANK = 32
LORA_ALPHA = 16
LORA_TARGET_MODULES = "all-linear"

SHARED_ADAPTER = "shared"
ORCHESTRATOR_ADAPTER = "orchestrator"
SELF_EDITOR_ADAPTER = "self_editor"  # IMP3 SEAL variant: writes strategy notes: see orchestration/seal/

_model = None
_tokenizer = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True


def _lora_config() -> LoraConfig:
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM")


def get_shared_model(device: str = "cuda:0"):
    """Returns (model, tokenizer), loading once and caching for the process.
    `model` carries the "shared", "orchestrator", and "self_editor" adapters;
    callers must call `use_adapter(model, name)` before generating/training to
    select which one is active."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    _model = get_peft_model(base, _lora_config(), adapter_name=SHARED_ADAPTER)
    _model.add_adapter(ORCHESTRATOR_ADAPTER, _lora_config())
    _model.add_adapter(SELF_EDITOR_ADAPTER, _lora_config())
    _model.set_adapter(SHARED_ADAPTER)
    return _model, _tokenizer


def use_adapter(model, adapter_name: str) -> None:
    model.set_adapter(adapter_name)


def set_adapter_trainable(model, adapter_name: str, trainable: bool) -> None:
    """Freeze/unfreeze exactly one adapter's LoRA params, leaving the other
    adapter and the base model untouched. Used by Baseline B to freeze the
    sub-agents' "shared" adapter while training only "orchestrator"."""
    for name, param in model.named_parameters():
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}"):
            param.requires_grad = trainable


@torch.no_grad()
def generate(
    model, tokenizer, system_prompt: str, user_prompt: str,
    cfg: Optional[GenerationConfig] = None, response_prefix: str = "",
) -> str:
    """`response_prefix`, if given (e.g. "<action>"), is appended to the
    rendered prompt text BEFORE generation, so the model continues from
    directly after it rather than needing to decide to emit the opening tag
    itself -- and is prepended back onto the returned string, so callers see
    exactly the same "raw model output" shape as before (parsing, recording,
    and training-time log-prob recomputation all stay unchanged).

    Added after runs/diagnostics/measure_malformed_rate.py found the untrained
    0.5B model ignored the required <action>/<switch> tag format on 100% of
    sampled turns even with a one-shot example in the system prompt (measured
    separately, not fixed by that alone) -- forcing the opening tag as a
    literal continuation is the standard, more reliable fix for a small
    instruct model that isn't yet reliably following a custom output schema."""
    cfg = cfg or GenerationConfig()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + response_prefix
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response_prefix + completion
