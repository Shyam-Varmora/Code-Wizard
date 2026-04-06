# TicketRouterEnv

OpenEnv environment that simulates **SaaS customer support triage**: for each ticket the agent chooses a **route** (engineering, billing, sales, trust_safety), a **priority** (P1–P4), and a **first-response SLA** (1h, 4h, 24h, 72h). Episodes queue **5–15** tickets (task-dependent). Episodes end when the queue is exhausted or when the **SLA failure budget** is exceeded.

## Problem

Support teams operate under **capacity limits** (hard task), **tier-dependent expectations** (enterprise vs free), and **SLA risk** if tickets are under-classified. Poor routing inflates queue depth, causes **SLA looseness violations**, and reduces a **customer satisfaction** proxy tracked in state.

## Environment design

- **Core logic** lives in `env/` (`models`, `data`, `reward`, `env`, `tasks`, `grader`).
- **OpenEnv HTTP / WebSocket server**: `server/app.py` uses `create_app(TicketRouterEnvironment, TicketRouterAction, TicketRouterObservation)` so hosted spaces expose **`POST /reset`**, **`POST /step`**, **`GET /state`**, **`GET /health`**, **`GET /schema`** in simulation mode.
- **Validation**: `openenv validate` (requires `uv.lock`; run `uv lock` after editing dependencies).

## Observation space (`TicketRouterObservation`)

| Field | Description |
| --- | --- |
| `ticket` | `{id, text, customer_tier}` — no hidden labels. |
| `team_load` | Integer **work units** per team after prior decisions. |
| `step_count` | Global step counter. |
| `sla_risk` | Heuristic **0–1** from public text, tier, and congestion (no label leakage). |
| `tickets_remaining` | Tickets left including the current one. |
| `task` | `ticket_router_{easy,medium,hard}` |
| `done`, `reward` | Gym-style termination and dense step reward. |
| `metadata.info` | Breakdown: SLA failures, overload events, reward factors, EMA satisfaction. |

## Action space (`TicketRouterAction`)

- `route`: `engineering` \| `billing` \| `sales` \| `trust_safety`
- `priority`: `P1` \| `P2` \| `P3` \| `P4`
- `sla`: `1h` \| `4h` \| `24h` \| `72h`

## Reward design (dense, non-binary)

Per step, `env/reward.py` combines:

- **Correctness**: piecewise credit for route (including **partial** credit for acceptable alternate teams on ambiguous tickets), priority closeness, and SLA closeness; scaled by **customer tier weighting**.
- **SLA violation penalty** when the promised SLA is **looser** than tier gold.
- **Team overload penalty** when post-step load exceeds per-team **capacity** (hard task).
- **Internal inconsistency penalty** (e.g., P1 with 72h).

Bonuses:

- **SLA over-delivery** (stricter SLA than required, capped).
- **Load-balancing bonus** when choosing the **least loaded** acceptable team.

Penalties:

- **Flip-flop** priority chaos across steps.
- **Escalation keyword** mismatch (urgent/legal/safety language with a relaxed priority).

## Tasks

| Task | File key | What makes it distinctive |
| --- | --- | --- |
| **Easy** | `ticket_router_easy` | Straightforward taxonomy; single gold route; generous capacity. |
| **Medium** | `ticket_router_medium` | Ambiguous tickets with **multiple acceptable routes**; **tier-conditioned** gold priority/SLA on several rows. |
| **Hard** | `ticket_router_hard` | **Tight per-team capacities**, longer episodes, and grader **efficiency** term (overload + balance across acceptable routes). |

Reset kwargs: pass `task="ticket_router_easy"` (etc.) inside the OpenEnv reset payload’s `data` object (WebSocket `{"type":"reset","data":{"task":"ticket_router_hard","seed":1}}`).

## Dataset

`env/data.py` defines **40** synthetic tickets spanning billing disputes, engineering incidents, enterprise sales cycles, trust & safety escalations, and cross-team edge cases. Each row stores **gold** labels used only for rewards and graders.

## Grading

`env/grader.py` exposes deterministic `grade_episode(task_name, trajectory)` → **`[0, 1]`**:

- **Route / priority / SLA** components with partial credit on alternates (medium/hard).
- **Hard task**: blends correctness with a **system efficiency** score derived from overload and intra-acceptable load spread.

## Setup

```bash
cd ticket-router-env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: uv sync
```

## Run the OpenEnv server

```bash
export PYTHONPATH="$(pwd)"
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Smoke the HTTP API:

```bash
curl -s -X POST http://127.0.0.1:8000/reset -H 'content-type: application/json' -d '{"task":"ticket_router_easy","seed":1}' | head
```

## Run baseline inference

`inference.py` uses the **`OpenAI` client** with `API_BASE_URL`, `MODEL_NAME`, and **`HF_TOKEN`** (or **`OPENAI_API_KEY`** if `HF_TOKEN` is unset). With a key set, each step calls the model for a **JSON** `{route,priority,sla}` and falls back to **deterministic rules** if parsing fails. Without a key, it runs **rules only** so CI/local runs stay offline.

**Default run:** `python inference.py` with **`TASK` unset** (or `TASK=all`) runs **easy → medium → hard** in order — three **`[START]` … `[END]`** blocks — matching the hackathon expectation of baseline scores on **all** graded tasks. Use `TASK=ticket_router_medium` (etc.) for a single episode.

**Hackathon log format:** lines must match the official sample — **two spaces** after `[STEP]`, and **`score=`** / **`rewards=`** at **two decimal places** on `[END]`:
`[END] success=true steps=6 score=0.92 rewards=0.34,1.00,...`

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=hf_...   # optional — rules-only baseline if unset
# All three tasks (default):
python inference.py
# Single task:
export TASK=ticket_router_hard
python inference.py
```

Expected log format:

```text
[START] task=<task> env=ticket_router model=<model>
[STEP]  step=<n> action=<json> reward=<float> done=<true|false> error=null
[END] success=<true|false> steps=<n> score=<0.00-1.00> rewards=<r1,r2,...>
```

`success` means the episode **cleared every ticket** without **early** termination from the SLA budget (mid-episode halt).

## Docker

```bash
docker build -t ticket-router-env .
docker run --rm -p 8000:8000 ticket-router-env
# Baseline (local rules; set HF_TOKEN to exercise OpenAI smoke):
docker run --rm -e TASK=ticket_router_easy -e HF_TOKEN= ticket-router-env python inference.py
```

## Baseline results (rule policy, seed=42, no network smoke)

Recorded on a reference CPU run (`HF_TOKEN` empty):

| Task | Steps | success | Notes |
| --- | ---: | :---: | --- |
| `ticket_router_easy` | 6 | true | Typical dense rewards 0.3–1.1 |
| `ticket_router_medium` | 12 | true | Ambiguity + tier labels stress SLA mapping |
| `ticket_router_hard` | 13 | true | Capacity pressure; occasional overload penalties |

Exact numbers vary slightly if curriculum sampling changes; re-run `python inference.py` (all tasks) or `TASK=ticket_router_hard python inference.py` for one task.

## Advanced features included

1. **Team capacity limits** — hard task only (`tasks.TASK_HARD.team_capacity`).
2. **SLA risk signal** — public heuristic in each observation (`sla_risk`).
3. **Customer tier weighting** — reward + grader emphasize enterprise accuracy.
4. **Load-balancing incentive** — reward bonus for choosing the least-loaded acceptable team.
5. **Escalation-aware penalties** — safety/legal language mismatched to relaxed priority is penalized.

## Pre-submission checklist (hackathon)

1. **`openenv validate`** passes from this directory (needs `uv.lock`; run `uv lock` after dependency edits).
2. **`docker build`** succeeds locally (validator uses repo root `Dockerfile` or `server/Dockerfile`).
3. **HF Space** URL responds **`POST /reset`** with **HTTP 200** (empty JSON body is fine for OpenEnv).
4. **`inference.py`** completes without error (default: all three tasks in one run), with your real **`HF_TOKEN`** / endpoint if you want LLM steps; finishes in **&lt; 20 minutes**; stdout uses **`[START]` / `[STEP]  ` / `[END]`** with **`score=`** on each **`[END]`** line.
5. Optional: `chmod +x scripts/validate-submission.sh && ./scripts/validate-submission.sh https://YOUR_SPACE.hf.space "$(pwd)"`

You can keep the problem statement / sample inference as reference under `docs/`; avoid committing large “Pasted text” files inside `env/` if you publish the package.

## License

This environment was authored for the Meta × PyTorch OpenEnv Hackathon; consult your submission’s license terms.
