# IMP2 Part 2 -- compare specialist: delegated to when the orchestrator wants
# candidate products compared/selected (and, if warranted, purchased). Shares
# weights/adapter with search_specialist by default (see model_pool.py).

from orchestration.sub_agent import SubAgentSpecialist

COMPARE_STEP_BUDGET = 5
COMPARE_INITIAL_SUBGOAL = (
    "Compare the currently visible product's attributes and price against the task requirements, "
    "and buy it now if it's a good match."
)


class CompareSpecialist(SubAgentSpecialist):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(role="compare", initial_subgoal=COMPARE_INITIAL_SUBGOAL, step_budget=COMPARE_STEP_BUDGET, device=device)
