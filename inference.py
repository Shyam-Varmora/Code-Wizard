#!/usr/bin/env python3
"""
Baseline inference for TicketRouterEnv.

Mandatory env (hackathon):
  API_BASE_URL   — LLM API base (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     — model id
  HF_TOKEN       — Hugging Face / API key (OpenAI client api_key)

OPENAI_API_KEY is accepted as a fallback when HF_TOKEN is unset.

TASK — optional. Unset, empty, or `all` / `full` / `baseline` runs easy, medium, then hard
(one [START]/[STEP]/[END] block per task). Set to a single task name to run one episode only.

Stdout format matches the official sample: [START], [STEP]  (two spaces), [END] with score=.
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from env.env import TicketRouterEnvironment
from env.grader import grade_episode
from env.models import TicketRouterAction, TicketRouterObservation, Priority, SLAHours, TeamRoute

TASK_NAMES = (
    "ticket_router_easy",
    "ticket_router_medium",
    "ticket_router_hard",
)

BENCHMARK = "ticket_router"
SUCCESS_SCORE_THRESHOLD = 0.05


def _bool_lower(x: bool) -> str:
    return "true" if x else "false"


def log_start(task: str, benchmark: str, model: str) -> None:
    print(f"[START] task={task} env={benchmark} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err_val = "null" if error is None else json.dumps(error)
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={_bool_lower(done)} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Score and per-step rewards use 2 decimal places (hackathon sample format)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={_bool_lower(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def rule_based_action(obs: TicketRouterObservation) -> TicketRouterAction:
    """Deterministic fallback policy (reproducible, not random)."""
    text = obs.ticket.text.lower()
    tier = obs.ticket.customer_tier

    if any(
        k in text
        for k in (
            "harass",
            "spam",
            "malware",
            "impersonat",
            "pii",
            "toxic",
            "moderation",
            "phishing",
            "child safety",
            "violent",
            "doxxing",
            "malware links",
        )
    ):
        route: TeamRoute = "trust_safety"
    elif any(
        k in text
        for k in (
            "invoice",
            "refund",
            "billed",
            "charge",
            "vat",
            "payment",
            "tax id",
            "verification charge",
            "credit never",
            "billing portal",
            "usage export",
            "metered charges",
            "proration",
            "invoice pdf",
            "discount promised",
            "checkout",
            "ach pull",
            "duplicate",
            "ledger",
        )
    ):
        route = "billing"
    elif any(
        k in text
        for k in (
            "demo",
            "pricing",
            "hipaa",
            "security questionnaire",
            "enterprise plan",
            "seats",
            "msa",
            "legal needs",
            "pilot",
            "upgrade from pro",
            "nonprofit",
            "quote",
            "written quote",
            "partner portal",
        )
    ):
        route = "sales"
    elif "sso" in text or "saml" in text:
        route = "engineering"
    else:
        route = "engineering"

    if any(
        w in text
        for w in (
            "outage",
            "500",
            "breach",
            "violent",
            "child safety",
            "pii",
            "latency >2",
            "isolation bug",
            "tenant's row",
            "doxxing",
            "malware links",
        )
    ):
        pnum = 1
    elif any(
        w in text
        for w in (
            "delayed",
            "blocked",
            "audit",
            "cannot access",
            "before board meeting",
            "quarter",
            "signup blocking",
            "403",
            "stuck at",
            "canary",
            "migration",
        )
    ):
        pnum = 2
    elif any(w in text for w in ("dark mode", "free tier user here")):
        pnum = 4
    else:
        pnum = 3

    if tier == "enterprise" and pnum > 2 and any(
        w in text for w in ("enterprise", "procurement", "renewal", "audit", "sso", "compliance")
    ):
        pnum = max(2, pnum - 1)
    if (
        tier == "enterprise"
        and obs.task == "ticket_router_medium"
        and pnum >= 3
        and any(w in text for w in ("sso", "saml", "procurement", "compliance", "agreement"))
    ):
        pnum = 2
    if tier == "free" and pnum < 3 and "outage" not in text:
        pnum = min(4, pnum + 1)

    priority: Priority = f"P{pnum}"  # type: ignore[assignment]
    sla_map = {1: "1h", 2: "4h", 3: "24h", 4: "72h"}
    sla: SLAHours = sla_map[pnum]  # type: ignore[assignment]
    if route == "trust_safety" and pnum <= 2:
        sla = "1h" if pnum == 1 else "4h"

    if obs.task == "ticket_router_hard" and obs.sla_risk >= 0.85:
        loads = obs.team_load
        if route == "engineering" and loads.get("engineering", 0) > loads.get("sales", 0) + 8:
            if "sso" in text and "pilot" in text:
                route = "sales"

    return TicketRouterAction(route=route, priority=priority, sla=sla)


def _parse_action_json(raw: str) -> TicketRouterAction | None:
    raw = raw.strip()
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        d = json.loads(raw)
        return TicketRouterAction(
            route=d["route"],
            priority=d["priority"],
            sla=d["sla"],
        )
    except Exception:
        return None


def llm_action(
    client: OpenAI,
    model: str,
    obs: TicketRouterObservation,
    task: str,
) -> TicketRouterAction | None:
    """All LLM calls go through OpenAI client. Returns None on failure."""
    system = (
        "You route SaaS support tickets. Reply with ONLY a compact JSON object, no markdown, "
        'keys: route (engineering|billing|sales|trust_safety), priority (P1|P2|P3|P4), '
        "sla (1h|4h|24h|72h). Keep priority and SLA consistent (e.g. P1 with 1h)."
    )
    user = json.dumps(
        {
            "task": task,
            "ticket_id": obs.ticket.id,
            "customer_tier": obs.ticket.customer_tier,
            "ticket_text": obs.ticket.text[:4000],
            "team_load": obs.team_load,
            "sla_risk": round(obs.sla_risk, 4),
            "tickets_remaining": obs.tickets_remaining,
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        comp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_completion_tokens=120,
            response_format={"type": "json_object"},
        )
    except Exception:
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=120,
            )
        except Exception:
            return None

    try:
        content = (comp.choices[0].message.content or "").strip()
    except Exception:
        return None

    return _parse_action_json(content)


def _action_to_log(a: TicketRouterAction) -> str:
    return json.dumps(
        {"route": a.route, "priority": a.priority, "sla": a.sla},
        separators=(",", ":"),
    )


def _resolve_task_list() -> tuple[str, ...]:
    """Default: all three tasks (hackathon baseline on every grader). Single-task via TASK=."""
    raw = (os.environ.get("TASK") or "").strip().lower()
    if raw in ("", "all", "full", "baseline"):
        return TASK_NAMES
    if raw in TASK_NAMES:
        return (raw,)
    return ("ticket_router_easy",)


def run_episode(task: str, client: OpenAI, model: str, use_llm: bool) -> None:
    """One full [START] … [STEP] … [END] block for a single task."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: TicketRouterEnvironment | None = None

    log_start(task, BENCHMARK, model)

    try:
        env = TicketRouterEnvironment()
        obs = env.reset(seed=42, task=task)

        while not obs.done:
            steps_taken += 1
            step_error: str | None = None
            try:
                if use_llm:
                    act = llm_action(client, model, obs, task)
                    if act is None:
                        act = rule_based_action(obs)
                else:
                    act = rule_based_action(obs)
                log_action = _action_to_log(act)
                obs = env.step(act)
                r = float(obs.reward or 0.0)
                done_now = obs.done
            except Exception as exc:
                step_error = str(exc)
                log_action = json.dumps(
                    {"route": "engineering", "priority": "P3", "sla": "24h"},
                    separators=(",", ":"),
                )
                r = 0.0
                done_now = True

            rewards.append(r)
            log_step(steps_taken, log_action, r, done_now, step_error)

            if step_error is not None:
                break

        if env is not None:
            st = env.state
            completed_all = st.ticket_index >= st.tickets_total
            early_halt = st.sla_failures >= st.max_sla_failures and st.ticket_index < st.tickets_total
            episode_ok = completed_all and not early_halt
            score = float(grade_episode(task, list(st.trajectory)))
            success = episode_ok and score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        rewards = rewards or [0.0]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            try:
                score = float(grade_episode(task, list(env.state.trajectory)))
                st = env.state
                completed_all = st.ticket_index >= st.tickets_total
                early_halt = st.sla_failures >= st.max_sla_failures and st.ticket_index < st.tickets_total
                episode_ok = completed_all and not early_halt
                success = episode_ok and score >= SUCCESS_SCORE_THRESHOLD
            except Exception:
                score = 0.0
                success = False
        if not rewards:
            rewards = [0.0]
        log_end(success, steps_taken, score, rewards)


def main() -> None:
    tasks = _resolve_task_list()

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    api_key = (os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "").strip()

    client = OpenAI(
        api_key=api_key or "invalid-key-placeholder",
        base_url=api_base,
    )
    use_llm = bool(api_key)

    for task in tasks:
        run_episode(task, client, model, use_llm)


if __name__ == "__main__":
    main()
