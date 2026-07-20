# IMP3 -- navigate specialist: delegated to when the orchestrator wants the
# task-relevant object/receptacle located (go to / open / examine). Ported
# from implementation-2's WebShop `search_specialist.py` -- same role in the
# two-phase structure (locate, then manipulate), reframed for ALFWorld's
# action space. Qwen2.5-0.5B-Instruct + LoRA, sharing weights with
# interact_specialist by default (see orchestration/model_pool.py).

from orchestration.sub_agent import SubAgentSpecialist

NAVIGATE_STEP_BUDGET = 5
NAVIGATE_INITIAL_SUBGOAL = (
    "Locate the object or receptacle the task requires -- go to it, opening or "
    "examining it as needed to find what you're looking for."
)


class NavigateSpecialist(SubAgentSpecialist):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(role="navigate", initial_subgoal=NAVIGATE_INITIAL_SUBGOAL, step_budget=NAVIGATE_STEP_BUDGET, device=device)
