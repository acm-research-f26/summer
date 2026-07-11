# IMP2 Part 2 -- search specialist: delegated to when the orchestrator wants
# products found/searched for. Qwen2.5-0.5B-Instruct + LoRA, sharing weights
# with compare_specialist by default (see orchestration/model_pool.py).

from orchestration.sub_agent import SubAgentSpecialist

SEARCH_STEP_BUDGET = 5
SEARCH_INITIAL_SUBGOAL = "Search for products matching the task, then open a promising result to view its details."


class SearchSpecialist(SubAgentSpecialist):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(role="search", initial_subgoal=SEARCH_INITIAL_SUBGOAL, step_budget=SEARCH_STEP_BUDGET, device=device)
