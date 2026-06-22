"""
representation.py
------------------
Implements the paper's "Adaptive Multi-Scale Network Representation"
(Section IV-C). Two scales:

  1. Full-Component Detail  - complete serialization, used for small
     networks (below a size threshold).
  2. Semantic Graph Abstraction - summarizes healthy regions, gives
     full detail only on violated components + their k-hop neighborhood
     of controllable assets. Used for larger networks / many violations.

The choice is made automatically based on network size and violation
count, exactly as described: "dynamically adjusts encoding granularity."
"""

from __future__ import annotations
import json
import pandapower.topology as top


FULL_DETAIL_BUS_THRESHOLD = 40     # networks at/below this size get full detail
FULL_DETAIL_VIOLATION_THRESHOLD = 8  # or below this many violations


def choose_scale(net, violations: list[dict]) -> str:
    n_buses = len(net.bus)
    if n_buses <= FULL_DETAIL_BUS_THRESHOLD or len(violations) <= FULL_DETAIL_VIOLATION_THRESHOLD:
        return "full"
    return "semantic_graph"


def _bus_summary(net, bus_idx: int) -> dict:
    row = net.bus.loc[bus_idx]
    summary = {"bus": int(bus_idx), "vn_kv": float(row.get("vn_kv", 0.0))}
    if "res_bus" in net and bus_idx in net.res_bus.index:
        summary["vm_pu"] = round(float(net.res_bus.at[bus_idx, "vm_pu"]), 4)
    return summary


def serialize_full(net, violations: list[dict], controllable) -> dict:
    """Full-Component Detail scale: serialize every bus, line, load."""
    buses = [_bus_summary(net, i) for i in net.bus.index]

    lines = []
    for i in net.line.index:
        row = net.line.loc[i]
        entry = {
            "line": int(i),
            "from_bus": int(row.from_bus), "to_bus": int(row.to_bus),
            "in_service": bool(row.in_service),
            "switchable": int(i) in controllable.switchable_lines,
        }
        if "res_line" in net and i in net.res_line.index:
            entry["loading_pct"] = round(float(net.res_line.at[i, "loading_percent"]), 2)
        lines.append(entry)

    loads = []
    for i in net.load.index:
        row = net.load.loc[i]
        loads.append({
            "load": int(i), "bus": int(row.bus), "p_mw": round(float(row.p_mw), 4),
            "curtailable": int(i) in controllable.curtailable_loads,
        })

    return {
        "scale": "full",
        "buses": buses,
        "lines": lines,
        "loads": loads,
        "violations": violations,
        "controllable_elements": {
            "switchable_lines": controllable.switchable_lines,
            "curtailable_loads": controllable.curtailable_loads,
            "battery_candidate_buses": controllable.battery_candidate_buses,
            "max_batteries": controllable.max_batteries,
        },
    }


def serialize_semantic_graph(net, violations: list[dict], controllable, hops: int = 1) -> dict:
    """
    Semantic Graph Abstraction scale: full detail only near violated
    components + nearby controllable assets; everything else summarized.
    """
    g = top.create_nxgraph(net)

    violated_buses = set()
    for v in violations:
        if v["type"] == "voltage" or v["type"] == "disconnected":
            violated_buses.add(v["id"])
        elif v["type"] == "thermal" and v.get("element", "line") == "line":
            row = net.line.loc[v["id"]]
            violated_buses.add(int(row.from_bus))
            violated_buses.add(int(row.to_bus))

    # expand by k hops to capture nearby controllable assets
    focus_buses = set(violated_buses)
    frontier = set(violated_buses)
    for _ in range(hops):
        nxt = set()
        for b in frontier:
            if b in g:
                nxt.update(g.neighbors(b))
        focus_buses.update(nxt)
        frontier = nxt

    detail_buses = [_bus_summary(net, b) for b in sorted(focus_buses) if b in net.bus.index]

    detail_lines = []
    for i in net.line.index:
        row = net.line.loc[i]
        if int(row.from_bus) in focus_buses or int(row.to_bus) in focus_buses or int(i) in controllable.switchable_lines:
            entry = {
                "line": int(i), "from_bus": int(row.from_bus), "to_bus": int(row.to_bus),
                "in_service": bool(row.in_service),
                "switchable": int(i) in controllable.switchable_lines,
            }
            if "res_line" in net and i in net.res_line.index:
                entry["loading_pct"] = round(float(net.res_line.at[i, "loading_percent"]), 2)
            detail_lines.append(entry)

    detail_loads = [
        {"load": int(i), "bus": int(net.load.at[i, "bus"]), "p_mw": round(float(net.load.at[i, "p_mw"]), 4),
         "curtailable": int(i) in controllable.curtailable_loads}
        for i in net.load.index if int(net.load.at[i, "bus"]) in focus_buses
    ]

    healthy_summary = {
        "n_buses_total": len(net.bus),
        "n_buses_in_focus": len(focus_buses),
        "n_lines_total": len(net.line),
        "n_lines_in_focus": len(detail_lines),
        "note": "Remaining buses/lines outside the focus set are within normal limits.",
    }

    return {
        "scale": "semantic_graph",
        "healthy_region_summary": healthy_summary,
        "focus_buses": detail_buses,
        "focus_lines": detail_lines,
        "focus_loads": detail_loads,
        "violations": violations,
        "controllable_elements": {
            "switchable_lines": controllable.switchable_lines,
            "curtailable_loads": controllable.curtailable_loads,
            "battery_candidate_buses": controllable.battery_candidate_buses,
            "max_batteries": controllable.max_batteries,
        },
    }


def build_representation(net, violations: list[dict], controllable) -> dict:
    """Top-level entry point: picks scale automatically, returns dict (JSON-serializable)."""
    scale = choose_scale(net, violations)
    if scale == "full":
        return serialize_full(net, violations, controllable)
    return serialize_semantic_graph(net, violations, controllable)


def representation_to_prompt_text(repr_dict: dict) -> str:
    """Render the representation dict as compact JSON text for the LLM prompt."""
    return json.dumps(repr_dict, separators=(",", ":"))
