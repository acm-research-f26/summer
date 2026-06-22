"""
actions.py
----------
Implements the action space from the paper (Section III-C):

  1. update_switch_status(line, closed: bool)   - Eq. (5)
  2. add_battery(bus, p_mw, q_mvar)              - Eqs. (8)-(11)
  3. curtail_load(load, gamma)                   - Eqs. (6)-(7)

Each action is validated against the network's ControllableElements
before being applied (the "Proactive Execution Check" of Sec IV-D is
implemented in orchestrator.py around calls to apply_action).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandapower as pp


class InvalidActionError(Exception):
    pass


@dataclass
class Action:
    tool: str            # "update_switch_status" | "add_battery" | "curtail_load"
    args: dict[str, Any]

    @staticmethod
    def from_dict(d: dict) -> "Action":
        if "tool" not in d or "args" not in d:
            raise InvalidActionError(f"Malformed action (missing tool/args): {d}")
        return Action(tool=d["tool"], args=d["args"])


def _check_battery_budget(net, controllable) -> int:
    """Count batteries already placed (as storage elements) to enforce Eq. (8)."""
    if "storage" not in net or len(net.storage) == 0:
        return 0
    return len(net.storage)


def apply_action(net, action: Action, controllable) -> str:
    """
    Applies a single action to `net` in place. Returns a short human-readable
    log string. Raises InvalidActionError if the action violates the action
    space constraints (this IS the "Proactive Execution Check", syntactic half).
    """
    if action.tool == "update_switch_status":
        line = int(action.args["line"])
        closed = bool(action.args["closed"])
        if line not in controllable.switchable_lines:
            raise InvalidActionError(f"Line {line} is not switchable.")
        if line not in net.line.index:
            raise InvalidActionError(f"Line {line} does not exist in network.")
        net.line.at[line, "in_service"] = closed
        return f"Set line {line} {'CLOSED' if closed else 'OPEN'}"

    elif action.tool == "add_battery":
        bus = int(action.args["bus"])
        p_mw = float(action.args.get("p_mw", 0.0))
        q_mvar = float(action.args.get("q_mvar", 0.0))

        if bus not in controllable.battery_candidate_buses:
            raise InvalidActionError(f"Bus {bus} is not eligible for battery placement.")
        if bus not in net.bus.index:
            raise InvalidActionError(f"Bus {bus} does not exist in network.")

        n_existing = _check_battery_budget(net, controllable)
        # if a battery already exists at this bus, treat as a dispatch update (Eq. 9-11)
        existing = None
        if "storage" in net and len(net.storage):
            match = net.storage.index[net.storage.bus == bus]
            if len(match):
                existing = match[0]

        if existing is None and n_existing >= controllable.max_batteries:
            raise InvalidActionError(
                f"Battery budget exceeded: {n_existing}/{controllable.max_batteries} already placed."
            )

        p_mw = max(-controllable.max_battery_p_mw, min(controllable.max_battery_p_mw, p_mw))
        q_mvar = max(-controllable.max_battery_q_mvar, min(controllable.max_battery_q_mvar, q_mvar))
        # enforce S^2 = P^2+Q^2 <= Smax^2 (Eq. 9), clip Q if needed
        s_max = (controllable.max_battery_p_mw ** 2 + controllable.max_battery_q_mvar ** 2) ** 0.5
        s = (p_mw ** 2 + q_mvar ** 2) ** 0.5
        if s > s_max and s > 0:
            scale = s_max / s
            p_mw *= scale
            q_mvar *= scale

        if existing is not None:
            net.storage.at[existing, "p_mw"] = -p_mw  # storage convention: positive=consumption
            net.storage.at[existing, "q_mvar"] = -q_mvar
            return f"Updated battery at bus {bus}: P={p_mw:.3f} MW, Q={q_mvar:.3f} MVAr"
        else:
            pp.create_storage(net, bus=bus, p_mw=-p_mw, q_mvar=-q_mvar,
                               max_e_mwh=1.0, soc_percent=50.0,
                               min_p_mw=-controllable.max_battery_p_mw,
                               max_p_mw=controllable.max_battery_p_mw,
                               name=f"battery_bus_{bus}")
            return f"Placed battery at bus {bus}: P={p_mw:.3f} MW, Q={q_mvar:.3f} MVAr"

    elif action.tool == "curtail_load":
        load = int(action.args["load"])
        gamma = float(action.args["gamma"])

        if load not in controllable.curtailable_loads:
            raise InvalidActionError(f"Load {load} is not curtailable.")
        if load not in net.load.index:
            raise InvalidActionError(f"Load {load} does not exist in network.")
        gamma = max(0.0, min(controllable.max_curtailment_fraction, gamma))

        if "_orig_p_mw" not in net.load.columns:
            net.load["_orig_p_mw"] = net.load["p_mw"]
            net.load["_orig_q_mvar"] = net.load["q_mvar"]

        orig_p = net.load.at[load, "_orig_p_mw"]
        orig_q = net.load.at[load, "_orig_q_mvar"]
        net.load.at[load, "p_mw"] = orig_p * (1 - gamma)
        net.load.at[load, "q_mvar"] = orig_q * (1 - gamma)
        return f"Curtailed load {load} by {gamma*100:.1f}%"

    else:
        raise InvalidActionError(f"Unknown action tool '{action.tool}'")


def action_space_schema(controllable) -> dict:
    """
    JSON schema-ish description handed to the LLM (paper Sec IV-B, item 3:
    "Action Space and Constraints").
    """
    return {
        "tools": [
            {
                "name": "update_switch_status",
                "description": "Open or close a switchable line to reconfigure topology.",
                "args": {"line": "int (must be one of switchable_lines)", "closed": "bool"},
            },
            {
                "name": "add_battery",
                "description": "Place or re-dispatch a battery at an eligible bus.",
                "args": {"bus": "int (must be one of battery_candidate_buses)",
                         "p_mw": f"float in [-{controllable.max_battery_p_mw}, {controllable.max_battery_p_mw}]",
                         "q_mvar": f"float in [-{controllable.max_battery_q_mvar}, {controllable.max_battery_q_mvar}]"},
            },
            {
                "name": "curtail_load",
                "description": "Reduce active/reactive power of a curtailable load by fraction gamma.",
                "args": {"load": "int (must be one of curtailable_loads)",
                         "gamma": f"float in [0, {controllable.max_curtailment_fraction}]"},
            },
        ],
        "constraints": {
            "max_batteries": controllable.max_batteries,
            "switchable_lines": controllable.switchable_lines,
            "curtailable_loads": controllable.curtailable_loads,
            "battery_candidate_buses": controllable.battery_candidate_buses,
        },
        "priority_policy": [
            "1. Topology Reconfiguration (update_switch_status) - prefer first, least disruptive",
            "2. Battery Deployment/Dispatch (add_battery) - prefer second",
            "3. Load Curtailment (curtail_load) - last resort, most disruptive",
        ],
    }
