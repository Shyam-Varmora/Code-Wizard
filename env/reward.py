"""Dense reward decomposition (non-binary) for ticket routing."""

from __future__ import annotations

from env.data import TicketRecord
from env.models import CustomerTier, Priority, SLAHours, TeamRoute, TicketRouterAction

_SLA_RANK: dict[SLAHours, int] = {"1h": 0, "4h": 1, "24h": 2, "72h": 3}


def _priority_weight(p: Priority) -> int:
    return {"P1": 4, "P2": 3, "P3": 2, "P4": 1}[p]


def _tier_route_weight(tier: CustomerTier) -> float:
    return {"enterprise": 1.4, "pro": 1.0, "free": 0.75}[tier]


def sla_strictness_gap(agent_sla: SLAHours, gold_sla: SLAHours) -> float:
    """Positive if agent promised looser SLA than required (violation severity in [0,1])."""
    ag = _SLA_RANK[agent_sla]
    gd = _SLA_RANK[gold_sla]
    if ag <= gd:
        return 0.0
    return min(1.0, (ag - gd) / 3.0)


def sla_better_than_gold(agent_sla: SLAHours, gold_sla: SLAHours) -> float:
    """Small bonus when agent over-delivers on SLA commitment."""
    ag = _SLA_RANK[agent_sla]
    gd = _SLA_RANK[gold_sla]
    if ag >= gd:
        return 0.0
    return min(0.15, (gd - ag) * 0.05)


def route_match_score(route: TeamRoute, acceptable: tuple[TeamRoute, ...]) -> float:
    if route in acceptable:
        if route == acceptable[0]:
            return 1.0
        return 0.85
    return 0.0


def priority_match_score(agent: Priority, gold: Priority) -> float:
    if agent == gold:
        return 1.0
    pa = int(agent[1])
    pg = int(gold[1])
    dist = abs(pa - pg)
    return max(0.0, 1.0 - 0.35 * dist)


def sla_match_score(agent: SLAHours, gold: SLAHours) -> float:
    if agent == gold:
        return 1.0
    return max(0.0, 1.0 - 0.4 * abs(_SLA_RANK[agent] - _SLA_RANK[gold]))


def internal_consistency_penalty(action: TicketRouterAction) -> float:
    """Penalize P1 with only 72h SLA, or P4 with 1h SLA (noisy policies)."""
    pr = int(action.priority[1])
    sr = _SLA_RANK[action.sla]
    pen = 0.0
    if pr <= 2 and sr >= 3:
        pen += 0.25
    if pr >= 4 and sr == 0:
        pen += 0.15
    return min(0.35, pen)


def escalation_keywords(text: str) -> bool:
    lower = text.lower()
    keys = (
        "outage",
        "legal",
        "breach",
        "child safety",
        "violent",
        "pii",
        "credit card",
        "malware",
        "harass",
    )
    return any(k in lower for k in keys)


def overload_penalty(
    team_load_after: dict[str, int],
    team_capacity: dict[str, int],
) -> float:
    """Fractional overload across teams, capped."""
    total = 0.0
    for team, cap in team_capacity.items():
        load = team_load_after.get(team, 0)
        if load > cap:
            total += (load - cap) / max(1, cap)
    return min(1.0, total * 0.35)


def load_balance_bonus(
    route: TeamRoute,
    team_load_before: dict[str, int],
    acceptable: tuple[TeamRoute, ...],
) -> float:
    """Small incentive to pick a less loaded acceptable team (advanced feature)."""
    if len(acceptable) < 2:
        return 0.0
    loads = [(t, team_load_before.get(t, 0)) for t in acceptable]
    loads_sorted = sorted(loads, key=lambda x: x[1])
    lightest = loads_sorted[0][0]
    if route == lightest:
        return 0.06
    if route == loads_sorted[1][0] and len(loads_sorted) > 1:
        return 0.03
    return 0.0


def compute_step_reward(
    *,
    record: TicketRecord,
    tier: CustomerTier,
    action: TicketRouterAction,
    team_load_before: dict[str, int],
    team_load_after: dict[str, int],
    team_capacity: dict[str, int],
    sla_violation: bool,
    last_route: str | None,
    last_priority: str | None,
) -> tuple[float, dict[str, float]]:
    """Return total reward and a breakdown dict (for info / debugging)."""
    gold_route = record.gold_route()
    acceptable = record.acceptable_routes
    gold_pri = record.gold_priority(tier)
    gold_sla = record.gold_sla(tier)
    tier_w = _tier_route_weight(tier)

    rr = route_match_score(action.route, acceptable)
    rp = priority_match_score(action.priority, gold_pri)
    rs = sla_match_score(action.sla, gold_sla)

    # Weight correctness slices; tier scales route/priority importance for enterprise.
    base = 0.38 * rr + 0.32 * rp + 0.30 * rs
    base *= 0.85 + 0.15 * tier_w

    gap = sla_strictness_gap(action.sla, gold_sla)
    viol_pen = 0.55 * gap + (0.35 if sla_violation else 0.0)
    viol_pen *= min(1.2, tier_w)

    over_pen = overload_penalty(team_load_after, team_capacity)
    inc = internal_consistency_penalty(action)

    # Flip-flop penalty: alternating P1/P4 on consecutive unrelated steps
    flip = 0.0
    if last_priority and last_route:
        if {last_priority, action.priority} == {"P1", "P4"} and action.route != last_route:
            flip = 0.08
    lb = load_balance_bonus(action.route, team_load_before, acceptable)
    extra_sla = sla_better_than_gold(action.sla, gold_sla)

    esc = escalation_keywords(record.text)
    esc_pen = 0.0
    if esc and action.priority in ("P3", "P4") and gold_pri in ("P1", "P2"):
        esc_pen = 0.12 * tier_w

    total = base - viol_pen - over_pen - inc - flip + lb + extra_sla - esc_pen
    total = max(-1.5, min(1.5, total))

    parts = {
        "base_correctness": base,
        "sla_penalty": viol_pen,
        "overload_penalty": over_pen,
        "consistency_penalty": inc,
        "flip_flop_penalty": flip,
        "load_balance_bonus": lb,
        "sla_overdelivery_bonus": extra_sla,
        "escalation_penalty": esc_pen,
    }
    return total, parts
