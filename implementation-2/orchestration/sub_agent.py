# IMP2 Part 2 -- shared Plan-Execute sub-agent loop used by both
# search_specialist.py and compare_specialist.py. The two roles differ only
# in their starting subgoal framing and step budget; the execution mechanism
# (Plan-Execute prompting, <switch>/<subgoal>/<action> parsing, credit via
# agents.sub_agent_credit at training time) is identical, so it lives here
# once rather than being duplicated per role.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_system.environments.env_package.webshop.projection import webshop_projection_options
from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE_OPTIONS,
    WEBSHOP_TEMPLATE_OPTIONS_NO_HIS,
)
from orchestration.model_pool import SHARED_ADAPTER, GenerationConfig, generate, use_adapter


def _format_available_actions(avail: Dict[str, Any]) -> str:
    actions = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in avail.get("clickables", []):
        actions.append(f"click[{txt}]")
    return "\n".join(f"'{a}'," for a in actions)


@dataclass
class SubAgentTurn:
    observation: str
    subgoal: str
    switch: str
    action: str
    valid: bool
    reward: float
    done: bool


@dataclass
class SubAgentResult:
    role: str
    turns: List[SubAgentTurn] = field(default_factory=list)
    final_observation: str = ""
    env_done: bool = False
    total_reward: float = 0.0


class SubAgentSpecialist:
    """Runs a short Plan-Execute sub-episode against a live WebShop env,
    starting from `initial_subgoal`, for up to `step_budget` turns or until
    the env signals done."""

    def __init__(self, role: str, initial_subgoal: str, step_budget: int, device: str = "cuda:0"):
        from orchestration.model_pool import get_shared_model

        self.role = role
        self.initial_subgoal = initial_subgoal
        self.step_budget = step_budget
        self.model, self.tokenizer = get_shared_model(device=device)

    def run(self, env, task_description: str, observation: str, avail_actions: Dict[str, Any]) -> SubAgentResult:
        result = SubAgentResult(role=self.role)
        current_subgoal = self.initial_subgoal

        for step in range(self.step_budget):
            prompt = WEBSHOP_TEMPLATE_OPTIONS_NO_HIS.format(
                task_description=task_description,
                current_observation=observation,
                current_subgoal=current_subgoal,
                available_actions=_format_available_actions(avail_actions),
            )
            use_adapter(self.model, SHARED_ADAPTER)
            raw = generate(self.model, self.tokenizer, system_prompt="", user_prompt=prompt, cfg=GenerationConfig())

            actions, subgoals, switches, valids = webshop_projection_options([raw])
            action, subgoal, switch, valid = actions[0], subgoals[0], switches[0], bool(valids[0])
            if switch == "SWITCH" and subgoal:
                current_subgoal = subgoal

            next_obs, reward, done, info = env.step(action)
            # WebAgentTextEnv.step() always returns info=None; available actions
            # come from a separate call, unlike the Ray-batched env manager's wrapper.
            avail_actions = env.get_available_actions() if not done else {}

            result.turns.append(SubAgentTurn(
                observation=observation, subgoal=current_subgoal, switch=switch,
                action=action, valid=valid, reward=float(reward), done=bool(done),
            ))
            result.total_reward += float(reward)
            observation = next_obs
            if done:
                result.env_done = True
                break

        result.final_observation = observation
        return result
