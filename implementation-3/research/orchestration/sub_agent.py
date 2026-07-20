# IMP3 -- shared Plan-Execute sub-agent loop used by both
# navigate_specialist.py and interact_specialist.py. The two roles differ only
# in their starting subgoal framing and step budget; the execution mechanism
# (Plan-Execute prompting, <switch>/<subgoal>/<action> parsing, credit via
# agents.sub_agent_credit at training time) is identical, so it lives here
# once rather than being duplicated per role.
#
# Ported from implementation-2's WebShop version: ALFWorld's admissible
# actions are a flat list (`info["admissible_commands"]`), not WebShop's
# `{has_search_bar, clickables}` page-model dict, so `_format_available_actions`
# and the type of `avail_actions` throughout this file changed accordingly.
# `alfworld_projection_options` also takes an extra `action_pools` argument
# that WebShop's single-arg `webshop_projection_options` doesn't.

from dataclasses import dataclass, field
from typing import List, Optional

from agent_system.environments.env_package.alfworld.projection import alfworld_projection_options
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE_OPTIONS_NO_HIS
from orchestration.model_pool import SHARED_ADAPTER, GenerationConfig, generate, use_adapter

# The vendored ALFWORLD_TEMPLATE_OPTIONS_NO_HIS (framework/agent_system/environments/
# prompts/alfworld.py) states the required <switch>/<subgoal>/<action> format as a
# spec but gives no worked example. Measured directly (runs/diagnostics/
# measure_malformed_rate.py): the untrained 0.5B model's raw output ignored the
# tag structure entirely on 100/100 sampled turns (e.g. dumping a bare list of
# admissible actions instead). Passed as a system-role message rather than
# edited into the vendored user-prompt template, so this one-shot fix stays
# scoped to this project's own code and doesn't touch the framework/ reference
# copy Paper Baseline also reads.
SUB_AGENT_SYSTEM_PROMPT = (
    "Example of a correctly formatted response (your own switch/subgoal/action content "
    "will differ, but it MUST follow exactly this structure, with no other text):\n"
    "<switch>SWITCH</switch>\n"
    "<subgoal>go to the fridge</subgoal>\n"
    "<action>go to fridge 1</action>"
)


def _format_available_actions(avail: List[str]) -> str:
    # matches AlfWorldEnvironmentManagerOptions.build_text_obs's own formatting
    # (env_manager.py), which the ALFWORLD_TEMPLATE_OPTIONS* prompts are written
    # against: quoted, newline+space-joined, 'help' filtered out.
    return "\n ".join(f"'{a}'" for a in avail if a != "help")


@dataclass
class SubAgentTurn:
    observation: str
    subgoal: str
    switch: str
    action: str
    valid: bool
    reward: float
    done: bool
    prompt: str = ""      # exact user prompt this turn's generation was conditioned on
    raw_output: str = ""  # exact raw model output (needed to reconstruct real token spans
                           # for training -- e.g. sub_agent_trainer.py's credit assignment
                           # -- a synthetic <switch>/<subgoal>/<action> reconstruction from
                           # the parsed fields would not match the actual generated tokens)


@dataclass
class SubAgentResult:
    role: str
    turns: List[SubAgentTurn] = field(default_factory=list)
    final_observation: str = ""
    env_done: bool = False
    total_reward: float = 0.0


class SubAgentSpecialist:
    """Runs a short Plan-Execute sub-episode against a live ALFWorld env,
    starting from `initial_subgoal`, for up to `step_budget` turns or until
    the env signals done."""

    def __init__(self, role: str, initial_subgoal: str, step_budget: int, device: str = "cuda:0"):
        from orchestration.model_pool import get_shared_model

        self.role = role
        self.initial_subgoal = initial_subgoal
        self.step_budget = step_budget
        self.model, self.tokenizer = get_shared_model(device=device)

    def run(self, env, task_description: str, observation: str, avail_actions: List[str]) -> SubAgentResult:
        result = SubAgentResult(role=self.role)
        current_subgoal = self.initial_subgoal

        for step in range(self.step_budget):
            prompt = ALFWORLD_TEMPLATE_OPTIONS_NO_HIS.format(
                current_observation=observation,
                current_subgoal=current_subgoal,
                admissible_actions=_format_available_actions(avail_actions),
            )
            use_adapter(self.model, SHARED_ADAPTER)
            raw = generate(self.model, self.tokenizer, system_prompt=SUB_AGENT_SYSTEM_PROMPT, user_prompt=prompt, cfg=GenerationConfig(), response_prefix="<switch>")

            actions, subgoals, switches, valids = alfworld_projection_options([raw], [avail_actions])
            action, subgoal, switch, valid = actions[0], subgoals[0], switches[0], bool(valids[0])
            if switch == "SWITCH" and subgoal:
                current_subgoal = subgoal

            next_obs, reward, done, info = env.step(action)
            avail_actions = env.get_available_actions() if not done else []

            result.turns.append(SubAgentTurn(
                observation=observation, subgoal=current_subgoal, switch=switch,
                action=action, valid=valid, reward=float(reward), done=bool(done),
                prompt=prompt, raw_output=raw,
            ))
            result.total_reward += float(reward)
            observation = next_obs
            if done:
                result.env_done = True
                break

        result.final_observation = observation
        return result
