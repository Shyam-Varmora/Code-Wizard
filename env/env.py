"""TicketRouterEnvironment — OpenEnv Environment implementation."""

from __future__ import annotations

import copy
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from env.data import TicketRecord, all_tickets
from env.models import (
    CustomerTier,
    TicketPublic,
    TicketRouterAction,
    TicketRouterObservation,
    TicketRouterState,
)
from env.reward import compute_step_reward, sla_strictness_gap
from env.tasks import DEFAULT_TASK, TASK_BY_NAME, TaskConfig, filter_pool

TEAMS: tuple[str, ...] = ("engineering", "billing", "sales", "trust_safety")


def _new_load_dict() -> dict[str, int]:
    return {t: 0 for t in TEAMS}


def _public_sla_risk(text: str, tier: CustomerTier, team_load: dict[str, int]) -> float:
    """Heuristic SLA risk without using hidden gold labels."""
    lower = text.lower()
    risk = 0.22
    if any(w in lower for w in ("down", "outage", "500", "latency", "breach", "pii", "legal", "safety", "violent", "malware")):
        risk += 0.38
    if any(w in lower for w in ("urgent", "asap", "today", "minutes", "blocked", "cannot")):
        risk += 0.12
    if tier == "enterprise":
        risk += 0.14
    elif tier == "free":
        risk -= 0.06
    max_load = max(team_load.values()) if team_load else 0
    risk += min(0.28, max_load / 55.0)
    return max(0.0, min(1.0, risk))


def _priority_load_units(p: str) -> int:
    return {"P1": 5, "P2": 4, "P3": 3, "P4": 2}[p]


def _build_episode_queue(
    rng: random.Random,
    cfg: TaskConfig,
    pool: list[TicketRecord],
) -> list[TicketRecord]:
    n = rng.randint(cfg.min_episode_len, cfg.max_episode_len)
    if not pool:
        raise RuntimeError("Empty ticket pool")
    if n <= len(pool):
        ixs = list(range(len(pool)))
        rng.shuffle(ixs)
        return [pool[i] for i in ixs[:n]]
    ixs = list(range(len(pool)))
    rng.shuffle(ixs)
    q = [pool[i] for i in ixs]
    need = n - len(q)
    q.extend(pool[rng.randrange(0, len(pool))] for _ in range(need))
    rng.shuffle(q)
    return q


class TicketRouterEnvironment(Environment[TicketRouterAction, TicketRouterObservation, TicketRouterState]):
    """Multi-ticket routing simulator with SLA, load, and tier-aware scoring."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = TicketRouterState()
        self._queue: list[TicketRecord] = []
        self._rng: random.Random = random.Random(0)
        self._task_cfg: TaskConfig = TASK_BY_NAME[DEFAULT_TASK]

    def _make_observation(
        self,
        *,
        record: TicketRecord | None,
        done: bool,
        reward: float | None,
        step_count: int,
    ) -> TicketRouterObservation:
        team_load = copy.deepcopy(dict(self._state.team_load))
        idx = self._state.ticket_index
        remaining = max(0, len(self._queue) - idx)
        if done or record is None:
            ticket = TicketPublic(
                id="__terminal__",
                text="",
                customer_tier="free",
            )
            rem = 0
        else:
            ticket = TicketPublic(
                id=record.id,
                text=record.text,
                customer_tier=record.customer_tier,
            )
            rem = len(self._queue) - idx
        sla_risk = _public_sla_risk(
            record.text if record else "",
            record.customer_tier if record else "free",
            team_load,
        )
        return TicketRouterObservation(
            ticket=ticket,
            team_load=team_load,
            step_count=step_count,
            sla_risk=sla_risk,
            tickets_remaining=rem,
            task=self._task_cfg.name,
            episode_id=self._state.episode_id,
            done=done,
            reward=reward,
            metadata={
                "info": {
                    "terminal": done,
                }
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TicketRouterObservation:
        self._reset_rubric()
        task_name = str(kwargs.get("task", DEFAULT_TASK))
        if task_name not in TASK_BY_NAME:
            task_name = DEFAULT_TASK
        self._task_cfg = TASK_BY_NAME[task_name]

        eid = episode_id or str(uuid4())
        s = seed
        if s is None:
            s = (hash(eid) & 0x7FFFFFFF) or 1
        self._rng = random.Random(s)

        pool = filter_pool(self._task_cfg.ticket_predicate, all_tickets())
        self._queue = _build_episode_queue(self._rng, self._task_cfg, pool)

        self._state = TicketRouterState(
            episode_id=eid,
            step_count=0,
            task=self._task_cfg.name,
            ticket_index=0,
            tickets_total=len(self._queue),
            team_load=_new_load_dict(),
            team_capacity=dict(self._task_cfg.team_capacity),
            sla_failures=0,
            max_sla_failures=self._task_cfg.max_sla_failures,
            overload_events=0,
            last_route=None,
            last_priority=None,
            customer_satisfaction_ema=1.0,
            trajectory=[],
            current_ticket_id=None,
            rng_seed=s,
        )
        cur = self._queue[0] if self._queue else None
        self._state.current_ticket_id = cur.id if cur else None
        return self._make_observation(
            record=cur,
            done=False,
            reward=0.0,
            step_count=0,
        )

    def step(
        self,
        action: TicketRouterAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketRouterObservation:
        del timeout_s, kwargs
        if self._state.ticket_index >= len(self._queue):
            return self._make_observation(
                record=None,
                done=True,
                reward=0.0,
                step_count=self._state.step_count,
            )

        record = self._queue[self._state.ticket_index]
        tier = record.customer_tier
        team_load_before = copy.deepcopy(dict(self._state.team_load))

        units = _priority_load_units(action.priority)
        self._state.team_load[action.route] = (
            self._state.team_load.get(action.route, 0) + units
        )
        team_load_after = copy.deepcopy(dict(self._state.team_load))

        cap = self._state.team_capacity
        overloaded = any(team_load_after.get(t, 0) > cap.get(t, 10**9) for t in TEAMS)
        if overloaded:
            self._state.overload_events += 1

        gap = sla_strictness_gap(action.sla, record.gold_sla(tier))
        sla_violation = gap > 0.0
        if sla_violation:
            self._state.sla_failures += 1

        reward, breakdown = compute_step_reward(
            record=record,
            tier=tier,
            action=action,
            team_load_before=team_load_before,
            team_load_after=team_load_after,
            team_capacity=cap,
            sla_violation=sla_violation,
            last_route=self._state.last_route,
            last_priority=self._state.last_priority,
        )

        self._state.trajectory.append(
            {
                "ticket_id": record.id,
                "customer_tier": tier,
                "route": action.route,
                "priority": action.priority,
                "sla": action.sla,
                "acceptable_routes": list(record.acceptable_routes),
                "team_load_after": team_load_after,
                "team_capacity": dict(cap),
                "overload": overloaded,
                "sla_strictness_gap": gap,
                "reward_breakdown": breakdown,
            }
        )

        self._state.customer_satisfaction_ema = max(
            0.0,
            min(
                1.0,
                0.72 * self._state.customer_satisfaction_ema
                + 0.28 * (0.5 + 0.5 * min(1.0, max(-1.0, reward))),
            ),
        )

        self._state.last_route = action.route
        self._state.last_priority = action.priority
        self._state.step_count += 1
        self._state.ticket_index += 1

        failed = self._state.sla_failures >= self._state.max_sla_failures
        finished = self._state.ticket_index >= len(self._queue)
        done = finished or failed

        if done:
            nxt = None
        elif self._state.ticket_index < len(self._queue):
            nxt = self._queue[self._state.ticket_index]
        else:
            nxt = None
        self._state.current_ticket_id = nxt.id if nxt else None

        obs = self._make_observation(
            record=nxt,
            done=done,
            reward=reward,
            step_count=self._state.step_count,
        )
        obs.metadata["info"] = {
            "terminal": done,
            "reason": "sla_budget" if failed and not finished else ("complete" if finished else "ok"),
            "sla_failures": self._state.sla_failures,
            "overload_events": self._state.overload_events,
            "reward_breakdown": breakdown,
            "customer_satisfaction_ema": self._state.customer_satisfaction_ema,
        }
        return obs

    @property
    def state(self) -> TicketRouterState:
        return self._state
