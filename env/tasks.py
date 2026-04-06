"""Task definitions: curriculum slices, episode sizing, and capacity constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.data import TicketRecord


@dataclass(frozen=True)
class TaskConfig:
    """Static configuration for a benchmark task."""

    name: str
    description: str
    min_episode_len: int
    max_episode_len: int
    max_sla_failures: int
    # Per-team capacity (hard task); easy/medium use high caps (no real constraint).
    team_capacity: dict[str, int]
    # Pool filter
    ticket_predicate: str  # name key; resolved in env


TASK_EASY = TaskConfig(
    name="ticket_router_easy",
    description="Clear routing keywords; single gold route; deterministic grades.",
    min_episode_len=6,
    max_episode_len=10,
    max_sla_failures=4,
    team_capacity={
        "engineering": 100,
        "billing": 100,
        "sales": 100,
        "trust_safety": 100,
    },
    ticket_predicate="easy",
)

TASK_MEDIUM = TaskConfig(
    name="ticket_router_medium",
    description="Ambiguous tickets with multiple valid teams; tier-dependent gold priority/SLA.",
    min_episode_len=7,
    max_episode_len=12,
    max_sla_failures=5,
    team_capacity={
        "engineering": 100,
        "billing": 100,
        "sales": 100,
        "trust_safety": 100,
    },
    ticket_predicate="medium",
)

TASK_HARD = TaskConfig(
    name="ticket_router_hard",
    description="Sequential routing under tight per-team capacity; trade off load, SLA risk, and correctness.",
    min_episode_len=8,
    max_episode_len=14,
    max_sla_failures=3,
    team_capacity={
        "engineering": 14,
        "billing": 12,
        "sales": 10,
        "trust_safety": 9,
    },
    ticket_predicate="hard",
)

TASK_BY_NAME: dict[str, TaskConfig] = {
    TASK_EASY.name: TASK_EASY,
    TASK_MEDIUM.name: TASK_MEDIUM,
    TASK_HARD.name: TASK_HARD,
}

DEFAULT_TASK = TASK_EASY.name


def filter_pool(
    predicate: str,
    all_tickets: tuple[TicketRecord, ...],
) -> list[TicketRecord]:
    """Return ordered pool for sampling (deterministic indices applied later)."""
    if predicate == "easy":
        return [t for t in all_tickets if t.category != "ambiguous"]
    if predicate == "medium":
        return [t for t in all_tickets if t.category == "ambiguous" or t.id.startswith("t_cap_")]
    if predicate == "hard":
        return [t for t in all_tickets if t.category in ("ambiguous", "engineering", "trust_safety") or t.id.startswith("t_cap_")]
    raise ValueError(f"Unknown ticket_predicate={predicate!r}")
