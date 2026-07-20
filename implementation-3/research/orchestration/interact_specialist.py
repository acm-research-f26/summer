# IMP3 -- interact specialist: delegated to when the orchestrator wants the
# located object acted on (take / put / heat / cool / clean / toggle). Ported
# from implementation-2's WebShop `compare_specialist.py` -- same role in the
# two-phase structure (locate, then manipulate), reframed for ALFWorld's
# action space. Shares weights/adapter with navigate_specialist by default
# (see model_pool.py).

from orchestration.sub_agent import SubAgentSpecialist

INTERACT_STEP_BUDGET = 5
INTERACT_INITIAL_SUBGOAL = (
    "Act on the object you've found -- take it, and put/heat/cool/clean/toggle "
    "it as the task requires."
)


class InteractSpecialist(SubAgentSpecialist):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(role="interact", initial_subgoal=INTERACT_INITIAL_SUBGOAL, step_budget=INTERACT_STEP_BUDGET, device=device)
