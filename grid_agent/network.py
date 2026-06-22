"""
network.py
----------
Loads benchmark networks (IEEE 30-bus, CIGRE MV) and implements the
violation model from the paper (Section III-A):

  V_volt  = buses with voltage outside [Vmin, Vmax]
  V_therm = lines/transformers with loading > 100%
  V_disc  = buses electrically isolated from the slack/external grid

Also defines each network's "controllable elements" (switchable lines,
curtailable loads, battery-eligible buses) and helpers to deliberately
inject violations, mirroring the paper's automated scenario generator
(Section V-A: Case30 Light/Medium, CIGRE MV Severe/Disconnected, etc.)
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top


VMIN, VMAX = 0.94, 1.06          # pu, voltage violation band
THERMAL_LIMIT_PCT = 100.0        # % loading, thermal violation threshold


@dataclass
class ControllableElements:
    """The action space available to the Planner (paper Sec III-C)."""
    switchable_lines: list[int] = field(default_factory=list)   # line indices
    curtailable_loads: list[int] = field(default_factory=list)  # load indices
    battery_candidate_buses: list[int] = field(default_factory=list)
    max_batteries: int = 3
    max_battery_p_mw: float = 0.5
    max_battery_q_mvar: float = 0.3
    max_curtailment_fraction: float = 0.5  # gamma_max


def load_network(case: str) -> tuple[Any, ControllableElements]:
    """
    Load one of the benchmark networks and define its controllable elements.

    case: "case30" | "cigre_mv"

    Returns (pandapower net, ControllableElements)
    """
    if case == "case30":
        net = pn.case30()
        ce = ControllableElements(
            switchable_lines=list(net.line.index[:6]),
            curtailable_loads=list(net.load.index),
            battery_candidate_buses=list(net.bus.index[:10]),
            max_batteries=2,
            max_battery_p_mw=10.0,
            max_battery_q_mvar=5.0,
        )
    elif case == "cigre_mv":
        net = pn.create_cigre_network_mv(with_der=False)
        ce = ControllableElements(
            switchable_lines=list(net.line.index),
            curtailable_loads=list(net.load.index),
            battery_candidate_buses=list(net.bus.index),
            max_batteries=3,
            max_battery_p_mw=0.5,
            max_battery_q_mvar=0.3,
        )
    else:
        raise ValueError(f"Unknown case '{case}'. Use 'case30' or 'cigre_mv'.")

    return net, ce


def run_power_flow(net) -> bool:
    """Run AC power flow. Returns True if converged, False otherwise."""
    try:
        pp.runpp(net, init="auto")
        return bool(net.get("converged", True))
    except Exception:
        return False


def detect_violations(net) -> list[dict]:
    """
    Implements Eqs. (1)-(2) plus disconnected-bus detection from the paper.
    Returns a list of violation dicts:
      {"type": "voltage"|"thermal"|"disconnected", "id": <bus/line idx>, ...}
    """
    violations: list[dict] = []

    converged = run_power_flow(net)
    if not converged:
        # Non-convergence usually itself signals a severe disconnection/instability.
        violations.append({"type": "non_convergence", "id": None,
                            "detail": "Power flow failed to converge."})
        # Still try to detect disconnected buses topologically.
        for b in top.unsupplied_buses(net):
            violations.append({"type": "disconnected", "id": int(b),
                                "detail": "Bus unsupplied from external grid."})
        return violations

    # Voltage violations (Eq. 1)
    for bus_idx, vm in net.res_bus.vm_pu.items():
        if vm < VMIN or vm > VMAX:
            violations.append({
                "type": "voltage", "id": int(bus_idx),
                "v_pu": round(float(vm), 4),
                "limits": [VMIN, VMAX],
            })

    # Thermal violations (Eq. 2), lines
    for line_idx, loading in net.res_line.loading_percent.items():
        if loading > THERMAL_LIMIT_PCT:
            violations.append({
                "type": "thermal", "id": int(line_idx), "element": "line",
                "loading_pct": round(float(loading), 2),
                "limit_pct": THERMAL_LIMIT_PCT,
            })

    # Thermal violations, transformers (if present)
    if "res_trafo" in net and len(net.res_trafo):
        for tr_idx, loading in net.res_trafo.loading_percent.items():
            if loading > THERMAL_LIMIT_PCT:
                violations.append({
                    "type": "thermal", "id": int(tr_idx), "element": "trafo",
                    "loading_pct": round(float(loading), 2),
                    "limit_pct": THERMAL_LIMIT_PCT,
                })

    # Disconnected buses
    for b in top.unsupplied_buses(net):
        violations.append({"type": "disconnected", "id": int(b),
                            "detail": "Bus unsupplied from external grid."})

    return violations


def is_resolved(violations: list[dict]) -> bool:
    return len(violations) == 0


def sandbox_copy(net):
    """Deep copy the network so the Executor can act without touching the original."""
    return copy.deepcopy(net)


# --- Scenario injection helpers, mirroring paper Sec V-A scenario names ---

def make_scenario(net, ce: ControllableElements, scenario: str):
    """
    Deliberately stresses the network to reproduce violation scenarios
    analogous to those in the paper (exact numeric severity will differ
    since the paper's injector isn't published, but the *mechanism*
    -- load scaling, line removal, bus isolation -- is the same idea).

    scenario: "light" | "medium" | "severe" | "disconnected"
    """
    net = sandbox_copy(net)

    if scenario == "light":
        net.load["p_mw"] *= 1.25
        net.load["q_mvar"] *= 1.25

    elif scenario == "medium":
        net.load["p_mw"] *= 1.45
        net.load["q_mvar"] *= 1.4

    elif scenario == "severe":
        net.load["p_mw"] *= 1.7
        net.load["q_mvar"] *= 1.6
        # also knock out a non-critical line to force thermal overloads elsewhere
        if len(ce.switchable_lines) > 1:
            net.line.at[ce.switchable_lines[-1], "in_service"] = False

    elif scenario == "disconnected":
        # open enough lines around a bus to isolate it from the slack
        victim_lines = ce.switchable_lines[: max(1, len(ce.switchable_lines) // 3)]
        for ln in victim_lines:
            net.line.at[ln, "in_service"] = False

    else:
        raise ValueError(f"Unknown scenario '{scenario}'")

    return net
