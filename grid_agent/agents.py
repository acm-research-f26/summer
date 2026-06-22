"""
agents.py
---------
The five specialized agents from the paper (Section IV-A):

  TopologyAgent   - parses state, runs power flow, finds violations
  PlannerAgent    - LLM-based reasoning engine, proposes action plans
  ExecutorAgent   - applies actions to a sandboxed network copy
  ValidatorAgent  - checks resolution + no new violations; rollback signal
  SummarizerAgent - human-readable explanation + continuous-learning record
"""

from __future__ import annotations
import json
import re

from .network import detect_violations, is_resolved
from .representation import build_representation, representation_to_prompt_text
from .actions import Action, apply_action, action_space_schema, InvalidActionError
from .llm_client import LLMClient


# ---------------------------------------------------------------------------
class TopologyAgent:
    """Initializes the resolution process (paper Sec IV-A)."""

    def analyze(self, net) -> list[dict]:
        return detect_violations(net)


# ---------------------------------------------------------------------------
class PlannerAgent:
    """
    The core reasoning engine (paper Sec IV-B). Builds the 5-part system
    prompt (role, state context, action space, strategic guidance, output
    schema) and asks the LLM for a JSON list of tool calls.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an expert power system operator responsible for resolving \
grid violations safely and efficiently.

ACTION SPACE AND CONSTRAINTS:
{action_schema}

STRATEGIC GUIDANCE (apply in this priority order):
1. Topology Reconfiguration (update_switch_status) - prefer first, least disruptive.
2. Battery Deployment/Dispatch (add_battery) - prefer second.
3. Load Curtailment (curtail_load) - last resort, most disruptive to customers.
Favor actions that resolve MULTIPLE violations at once (coordinated, system-level \
solutions) over redundant, scattered single-violation fixes.

OUTPUT SCHEMA:
Respond with ONLY a JSON array of tool calls, and nothing else (no prose, no \
markdown fences). Each element must look like:
  {{"tool": "<tool_name>", "args": {{...}}}}
Example:
[{{"tool": "update_switch_status", "args": {{"line": 12, "closed": false}}}},
 {{"tool": "add_battery", "args": {{"bus": 5, "p_mw": 0.3, "q_mvar": 0.1}}}}]
"""

    USER_PROMPT_TEMPLATE = """CURRENT NETWORK STATE AND VIOLATIONS (JSON):
{state_json}

Propose a coordinated action plan (a JSON array of tool calls) that resolves \
as many violations as possible using as few actions as possible, respecting \
all constraints above."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, net, violations: list[dict], controllable) -> list[Action]:
        repr_dict = build_representation(net, violations, controllable)
        state_json = representation_to_prompt_text(repr_dict)

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            action_schema=json.dumps(action_space_schema(controllable), indent=None)
        )
        user_prompt = self.USER_PROMPT_TEMPLATE.format(state_json=state_json)

        raw = self.llm.complete(system_prompt, user_prompt)
        return self._parse_actions(raw)

    @staticmethod
    def _parse_actions(raw: str) -> list[Action]:
        # Strip markdown fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Planner returned non-JSON output: {raw!r}") from e
        if not isinstance(data, list):
            raise ValueError(f"Planner output must be a JSON array, got: {type(data)}")
        return [Action.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
class ExecutorAgent:
    """
    Interfaces to the power flow solver, applying actions to a sandboxed
    copy of the network (paper Sec IV-A and the "Proactive Execution Check"
    of Sec IV-D).
    """

    def execute(self, net, actions: list[Action], controllable):
        """
        Applies actions one by one to `net` (already a sandbox). Stops early
        if violations are already resolved (Algorithm 1, lines 9-14).
        Returns (logs, violations_after).
        """
        from .network import run_power_flow

        logs = []
        for action in actions:
            try:
                msg = apply_action(net, action, controllable)
                logs.append(msg)
            except InvalidActionError as e:
                logs.append(f"REJECTED action {action.tool}({action.args}): {e}")
                continue

            converged = run_power_flow(net)
            if not converged:
                logs.append("WARNING: power flow did not converge after this action.")
                continue

            current_violations = detect_violations(net)
            if is_resolved(current_violations):
                logs.append("All violations resolved; stopping early.")
                return logs, current_violations

        final_violations = detect_violations(net)
        return logs, final_violations


# ---------------------------------------------------------------------------
class ValidatorAgent:
    """
    Post-hoc state assessment + monotonic progress guarantee (paper Sec IV-D).
    """

    @staticmethod
    def _violation_score(violations: list[dict]) -> float:
        """Simple severity score: count, weighted slightly by type."""
        weights = {"non_convergence": 5.0, "disconnected": 3.0, "voltage": 1.0, "thermal": 1.0}
        return sum(weights.get(v["type"], 1.0) for v in violations)

    def validate(self, violations_before: list[dict], violations_after: list[dict]) -> dict:
        resolved = is_resolved(violations_after)
        score_before = self._violation_score(violations_before)
        score_after = self._violation_score(violations_after)
        improved = score_after < score_before

        before_keys = {(v["type"], v.get("id")) for v in violations_before}
        new_violations = [v for v in violations_after if (v["type"], v.get("id")) not in before_keys]

        return {
            "resolved": resolved,
            "improved": improved,
            "score_before": score_before,
            "score_after": score_after,
            "new_violations": new_violations,
            "accept": resolved or (improved and len(new_violations) == 0),
        }


# ---------------------------------------------------------------------------
class SummarizerAgent:
    """Explainability + continuous-learning data record (paper Sec IV-A, IV-E)."""

    def summarize(self, initial_violations: list[dict], action_log: list[dict],
                  final_violations: list[dict], success: bool) -> dict:
        explanation_lines = ["Grid-Agent resolution summary:",
                             f"- Initial violations ({len(initial_violations)}): "
                             f"{[(v['type'], v.get('id')) for v in initial_violations]}"]
        for i, entry in enumerate(action_log, 1):
            explanation_lines.append(f"- Iteration {i}: {entry['logs']}")
            explanation_lines.append(
                f"    validator: resolved={entry['validation']['resolved']} "
                f"improved={entry['validation']['improved']} "
                f"accepted={entry['validation']['accept']}"
            )
        explanation_lines.append(
            f"- Final result: {'SUCCESS - all violations resolved' if success else 'INCOMPLETE'} "
            f"({len(final_violations)} violations remaining)."
        )
        explanation = "\n".join(explanation_lines)

        training_record = {
            "initial_violations": initial_violations,
            "action_sequence": action_log,
            "final_violations": final_violations,
            "success": success,
            "explanation": explanation,
        }
        return training_record
