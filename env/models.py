"""Pydantic models for actions, observations, and environment state."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

TeamRoute = Literal["engineering", "billing", "sales", "trust_safety"]
Priority = Literal["P1", "P2", "P3", "P4"]
SLAHours = Literal["1h", "4h", "24h", "72h"]
CustomerTier = Literal["free", "pro", "enterprise"]


class TicketPublic(BaseModel):
    """Ticket fields visible to the agent (no gold labels)."""

    id: str = Field(..., description="Stable ticket identifier")
    text: str = Field(..., description="Customer message")
    customer_tier: CustomerTier = Field(..., description="Support tier")


class TicketRouterAction(Action):
    """Routing decision for the current ticket."""

    route: TeamRoute = Field(..., description="Target team queue")
    priority: Priority = Field(..., description="Priority tier")
    sla: SLAHours = Field(..., description="Committed first-response SLA")


class TicketRouterObservation(Observation):
    """What the agent sees before deciding on the current ticket."""

    ticket: TicketPublic = Field(..., description="Current ticket")
    team_load: dict[str, int] = Field(
        ...,
        description="Open work units per team (simulated queue depth)",
    )
    step_count: int = Field(..., ge=0, description="Environment step index")
    sla_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated risk of SLA breach if mishandled (0=low, 1=high)",
    )
    tickets_remaining: int = Field(
        ...,
        ge=0,
        description="Tickets left in this episode including the current one",
    )
    task: str = Field(..., description="Active curriculum task id")
    episode_id: str | None = Field(
        default=None,
        description="Opaque episode id for logging",
    )


class TicketRouterState(State):
    """Internal simulator state (may include fields beyond base State)."""

    task: str = Field(default="easy", description="Task name")
    episode_id: str | None = Field(default=None)
    ticket_index: int = Field(default=0, ge=0)
    tickets_total: int = Field(default=0, ge=0)
    team_load: dict[str, int] = Field(default_factory=dict)
    team_capacity: dict[str, int] = Field(default_factory=dict)
    sla_failures: int = Field(default=0, ge=0)
    max_sla_failures: int = Field(default=3, ge=1)
    overload_events: int = Field(default=0, ge=0)
    last_route: str | None = Field(default=None)
    last_priority: str | None = Field(default=None)
    customer_satisfaction_ema: float = Field(default=1.0)
    # Serialized trajectory for graders / analysis
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    current_ticket_id: str | None = Field(default=None)
    rng_seed: int | None = Field(default=None)
