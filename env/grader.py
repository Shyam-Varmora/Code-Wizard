"""Deterministic episode graders (0.0–1.0) per task."""

from __future__ import annotations

from typing import Any

from env.data import TicketRecord, all_tickets
from env.models import CustomerTier
from env.reward import (
    overload_penalty,
    priority_match_score,
    route_match_score,
    sla_match_score,
    sla_strictness_gap,
)
from env.tasks import TASK_BY_NAME


def _ticket_map() -> dict[str, TicketRecord]:
    return {t.id: t for t in all_tickets()}


def grade_episode(
    task_name: str,
    trajectory: list[dict[str, Any]],
    final_state: dict[str, Any] | None = None,
) -> float:
    """
    Deterministic scalar grade aggregating routing quality and (for hard) efficiency.

    Trajectory items expect keys:
    ticket_id, customer_tier, route, priority, sla,
    team_load_after (dict), team_capacity (dict), acceptable_routes (list|tuple)
    """
    if task_name not in TASK_BY_NAME:
        raise ValueError(f"Unknown task {task_name}")
    task = TASK_BY_NAME[task_name]
    tmap = _ticket_map()
    if not trajectory:
        return 0.0

    scores: list[float] = []
    eff_scores: list[float] = []

    for step in trajectory:
        tid = step["ticket_id"]
        tier = step["customer_tier"]
        if tid not in tmap:
            continue
        rec = tmap[tid]
        tier_t: CustomerTier = tier  # type: ignore[assignment]
        gold_pri = rec.gold_priority(tier_t)
        gold_sla = rec.gold_sla(tier_t)
        acceptable_raw = step.get("acceptable_routes", rec.acceptable_routes)
        acceptable = tuple(acceptable_raw)
        route = step["route"]
        priority = step["priority"]
        sla = step["sla"]

        r_route = route_match_score(route, acceptable)
        r_pri = priority_match_score(priority, gold_pri)
        r_sla = sla_match_score(sla, gold_sla)
        gap = sla_strictness_gap(sla, gold_sla)

        per = 0.42 * r_route + 0.33 * r_pri + 0.25 * r_sla
        per *= max(0.0, 1.0 - gap * 0.9)
        scores.append(per)

        if task.name == "ticket_router_hard":
            cap = step.get("team_capacity") or task.team_capacity
            load_after = step.get("team_load_after") or {}
            eff = 1.0 - overload_penalty(load_after, cap)
            loads = [load_after.get(t, 0) for t in acceptable]
            hi = max(loads) if loads else 0
            lo = min(loads) if loads else 0
            spread = (hi - lo) / max(1, hi)
            eff = eff * (0.85 + 0.15 * (1.0 - min(1.0, spread * 0.5)))
            eff_scores.append(max(0.0, min(1.0, eff)))

    if not scores:
        return 0.0

    if task.name == "ticket_router_easy":
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    if task.name == "ticket_router_medium":
        avg = sum(scores) / len(scores)
        pri_variance_pen = 0.0
        tiers_seen: dict[str, list[float]] = {}
        for step, s in zip(trajectory, scores):
            tr = str(step["customer_tier"])
            tiers_seen.setdefault(tr, []).append(s)
        for vals in tiers_seen.values():
            if len(vals) > 1:
                m = sum(vals) / len(vals)
                pri_variance_pen += sum(abs(v - m) for v in vals) / len(vals) * 0.02
        return max(0.0, min(1.0, avg - min(0.08, pri_variance_pen)))

    route_quality = sum(scores) / len(scores)
    efficiency = sum(eff_scores) / len(eff_scores) if eff_scores else 1.0
    return max(0.0, min(1.0, 0.72 * route_quality + 0.28 * efficiency))


def grade_from_state_stub(state_dump: dict[str, Any]) -> float:
    task = state_dump.get("task", "ticket_router_easy")
    traj = state_dump.get("trajectory", [])
    return grade_episode(task, traj, state_dump)


def public_task_list() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for key in sorted(TASK_BY_NAME.keys()):
        cfg = TASK_BY_NAME[key]
        out.append({"name": cfg.name, "description": cfg.description})
    return out
