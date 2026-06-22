"""
orchestrator.py
----------------
Implements Algorithm 1 from the paper: the cyclic
plan -> execute -> validate -> (accept | rollback) loop, with the
Topology and Summarizer agents bookending the process.
"""

from __future__ import annotations
import copy
import logging

from .network import detect_violations, is_resolved, run_power_flow
from .agents import TopologyAgent, PlannerAgent, ExecutorAgent, ValidatorAgent, SummarizerAgent
from .llm_client import LLMClient

log = logging.getLogger("grid_agent")


class GridAgentOrchestrator:
    def __init__(self, llm: LLMClient, max_iterations: int = 5):
        self.topology_agent = TopologyAgent()
        self.planner_agent = PlannerAgent(llm)
        self.executor_agent = ExecutorAgent()
        self.validator_agent = ValidatorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.max_iterations = max_iterations

    def run(self, net, controllable, verbose: bool = True) -> dict:
        """
        Implements Algorithm 1:
          Nwork <- Initialize sandboxed environment(N)
          V <- Analyze violations(Nwork)
          while not IsResolved(V) and iter < Tmax:
              C <- Generate Network representation(Nwork, V)
              A <- LLM planning(C, Aavailable)
              for action in A: Execute action; check resolution; break if resolved
              effectiveness <- Evaluate actions(...)
              V <- Vnew
          explanation <- Generate explanation(...)
          return Nwork, explanation
        """
        run_power_flow(net)  # establish initial res_* tables
        n_work = copy.deepcopy(net)  # Step 1: sandboxed environment

        initial_violations = self.topology_agent.analyze(n_work)
        violations = initial_violations
        action_log = []
        iteration = 0

        if verbose:
            print(f"[TopologyAgent] Initial violations: {len(violations)}")

        while not is_resolved(violations) and iteration < self.max_iterations:
            iteration += 1
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
                print(f"[Planner] Planning for {len(violations)} violations...")

            # Snapshot for rollback (Monotonic Progress Guarantee, Sec IV-D)
            n_before = copy.deepcopy(n_work)
            violations_before = violations

            try:
                actions = self.planner_agent.plan(n_work, violations, controllable)
            except Exception as e:
                log.warning(f"Planner failed to produce a valid plan: {e}")
                action_log.append({"iteration": iteration, "logs": [f"PLANNER ERROR: {e}"],
                                    "validation": {"resolved": False, "improved": False, "accept": False}})
                break

            if verbose:
                print(f"[Planner] Proposed {len(actions)} action(s): "
                      f"{[(a.tool, a.args) for a in actions]}")

            logs, violations_after = self.executor_agent.execute(n_work, actions, controllable)
            if verbose:
                for l in logs:
                    print(f"[Executor] {l}")

            validation = self.validator_agent.validate(violations_before, violations_after)
            if verbose:
                print(f"[Validator] resolved={validation['resolved']} "
                      f"improved={validation['improved']} accept={validation['accept']}")

            if not validation["accept"]:
                # Rollback (Sec IV-D)
                n_work = n_before
                run_power_flow(n_work)
                violations_after = detect_violations(n_work)
                if verbose:
                    print("[Validator] Plan rejected -> ROLLED BACK to previous state.")

            action_log.append({
                "iteration": iteration,
                "actions": [{"tool": a.tool, "args": a.args} for a in actions],
                "logs": logs,
                "validation": validation,
            })

            violations = violations_after

        success = is_resolved(violations)
        record = self.summarizer_agent.summarize(initial_violations, action_log, violations, success)

        if verbose:
            print("\n" + record["explanation"])

        return {
            "success": success,
            "final_network": n_work,
            "final_violations": violations,
            "iterations": iteration,
            "action_log": action_log,
            "training_record": record,
        }
