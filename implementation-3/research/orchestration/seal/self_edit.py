# IMP3 SEAL variant -- turns a strategy note into a small LoRA training
# update (SEAL's "self-edit becomes synthetic training data via the model's
# own generations" mechanism, filtered by real outcome rather than inventing
# corrections for failures).
#
# CORRECTNESS-CRITICAL: the fine-tune step here must update the "shared"
# adapter (both sub-agent specialists), NOT the "self_editor" adapter. The
# note's whole purpose is to change *sub-agent behavior*; self_editor is only
# used to *generate* the note (see note_writer.py). Getting this backwards
# would silently make the self-edit step a no-op on sub-agent behavior while
# corrupting the note-writer's own weights outside its RL update path. See
# `run_self_edit_step`'s explicit `set_adapter_trainable(model, SHARED_ADAPTER, ...)`
# call below, and the dedicated test in seal/tests (or ad hoc assertion) that
# only SHARED_ADAPTER-named params get nonzero grad.

from typing import Dict, List

import torch
import torch.nn.functional as F

from orchestration.model_pool import SHARED_ADAPTER, set_adapter_trainable, use_adapter
from orchestration.sub_agent import SUB_AGENT_SYSTEM_PROMPT

MIN_EXAMPLES_TO_APPLY = 1  # below this, a self-edit step is skipped (nothing to learn from)


def note_to_training_examples(note: str, episode_logs, max_examples: int = 8) -> List[Dict[str, str]]:
    """Builds a small synthetic SFT set from a strategy note plus recent
    episode transcripts: note-prefixed prompts paired with the model's own
    responses from SUCCESSFUL episode turns only. Turns from failed episodes
    are skipped rather than fabricating a "corrected" action -- inventing an
    unverified correction would not be faithful to SEAL's actual approach of
    using the model's own (outcome-filtered) generations as training data."""
    if not note:
        return []  # malformed/empty note from write_note() -> no-op edit

    examples: List[Dict[str, str]] = []
    for log in episode_logs:
        if not log.success:
            continue
        for round_ in log.rounds:
            result = round_.sub_agent_result
            if result is None:
                continue
            for t in result["turns"]:
                examples.append({
                    "prompt": f"Recent strategy note: {note}\n\n{t['prompt']}",
                    "target": t["raw_output"],
                    # must match the system prompt sub_agent.py actually generated
                    # `t["raw_output"]` under -- see run_self_edit_step's use of this.
                    "system_prompt": SUB_AGENT_SYSTEM_PROMPT,
                })
                if len(examples) >= max_examples:
                    return examples
    return examples


def snapshot_adapter(model, adapter_name: str) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}")
    }


def restore_adapter(model, adapter_name: str, snapshot: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name])


def run_self_edit_step(model, tokenizer, examples: List[Dict[str, str]], lr: float = 5e-5, num_steps: int = 4) -> dict:
    """A handful of gradient steps of plain causal-LM SFT on `examples`,
    restricted to the SHARED_ADAPTER's params. Returns per-step loss for
    logging; a no-op (zero steps taken) if `examples` is too sparse to be
    worth a gradient step -- report that honestly rather than forcing a
    step on padding/fabricated data."""
    if len(examples) < MIN_EXAMPLES_TO_APPLY:
        return {"steps_taken": 0, "losses": [], "reason": "too few examples"}

    for p in model.parameters():
        p.requires_grad = False
    set_adapter_trainable(model, SHARED_ADAPTER, True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found on the shared adapter -- check adapter naming.")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    use_adapter(model, SHARED_ADAPTER)

    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        step_losses = []
        for ex in examples:
            messages = [{"role": "system", "content": ex.get("system_prompt", "")}, {"role": "user", "content": ex["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
            target_ids = tokenizer(ex["target"], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if target_ids.shape[1] == 0:
                continue
            input_ids = torch.cat([prompt_ids, target_ids], dim=1)

            logits = model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            target_len = target_ids.shape[1]
            loss = -token_logprobs[0, -target_len:].mean() / len(examples)
            loss.backward()
            step_losses.append(loss.item())
        optimizer.step()
        losses.append(sum(step_losses))

    return {"steps_taken": num_steps, "losses": losses, "num_examples": len(examples)}
