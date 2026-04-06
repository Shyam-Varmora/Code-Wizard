"""FastAPI application for TicketRouterEnv (OpenEnv create_app)."""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

from env.env import TicketRouterEnvironment
from env.models import TicketRouterAction, TicketRouterObservation

app = create_app(
    TicketRouterEnvironment,
    TicketRouterAction,
    TicketRouterObservation,
    env_name=os.getenv("OPENENV_NAME", "ticket_router"),
)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
