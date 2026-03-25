---
title: OpenEnv Support Triage
emoji: 📬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - agent-eval
  - reinforcement-learning
---

# OpenEnv Support Triage

A real-world OpenEnv environment for **customer support inbox triage**.

This environment simulates work done daily by support operations teams: reading incoming tickets, classifying issue type, assigning to the right team, drafting policy-safe responses, and closing tickets only when ready.

## Why this is real-world useful

Support triage is a high-impact operational bottleneck for SaaS and e-commerce teams. This environment helps evaluate whether an AI agent can:

- route issues accurately,
- respect urgency/SLA signals,
- avoid unsafe promises in customer communications,
- and complete a full multi-step workflow under step limits.

## OpenEnv spec compliance

Implements typed Pydantic models and standard API:

- `Observation`: `openenv_support_triage.models.Observation`
- `Action`: `openenv_support_triage.models.Action`
- `Reward`: `openenv_support_triage.models.Reward`

Core methods exposed via HTTP:

- `POST /reset` → initial observation
- `POST /step` → `{ observation, reward, done, info }`
- `GET /state` → current typed state

Metadata is in `openenv.yaml`.

## Action and observation spaces

### Observation

Each observation includes:

- task metadata (`task_id`, `task_name`, `difficulty`),
- progression (`step_count`, `max_steps`),
- ticket list with fields:
  - `ticket_id`, `customer`, `subject`, `body`
  - `priority` (1–5), `sla_hours`
  - agent-updated fields: `predicted_category`, `assigned_team`, `reply_draft`, `closed`
- `recent_actions` (loop awareness)
- `guidance`

### Action

`action_type` in:

- `classify`
- `assign`
- `draft_reply`
- `close`
- `noop`

Additional fields:

- `ticket_id` (required for all except noop)
- `value` (required for classify/assign)
- `message` (required for draft_reply)

See `GET /tasks` for machine-readable action schema.

## Task set (easy → hard)

1. **support-easy-001** — Billing + Login Triage
   - 2 straightforward tickets
   - tests basic classify/assign/reply/close pipeline

2. **support-medium-001** — Priority SLA Handling
   - 3 tickets with VIP + SLA pressure + security lockout
   - tests prioritization and policy-safe communication

3. **support-hard-001** — Fraud, Policy, and Escalation
   - 4 high-stakes tickets (fraud, data exposure, exception policy)
   - tests nuanced escalation and non-destructive behavior

## Deterministic grader (0.0–1.0)

Per ticket score:

- 25% classification correctness
- 25% assignment correctness
- 25% reply quality (required keywords / forbidden language)
- 25% closure validity (closed only after sufficient triage quality)

Task score = average ticket score in `[0.0, 1.0]`.

`GET /grader` returns current task score at any time and final score when done.

## Reward shaping

Dense trajectory reward (not sparse-only):

- step cost: `-0.01`
- correct classify/assign: positive reward
- incorrect classify/assign: negative reward
- draft quality improvements: incremental positive signal
- high-quality reply bonus
- premature/invalid/destructive operations: penalties
- repeated-loop behavior: penalty
- all tickets properly closed: completion bonus
- max-step timeout: penalty

This gives useful partial progress signal while discouraging loops/destructive actions.

## API endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `POST /baseline`

## Local setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Create a persistent env file (loaded automatically on every run):

```bash
# Windows PowerShell
Copy-Item .env.example .env
# edit .env and set OPENAI_API_KEY=...
```

Health check:

```bash
curl http://localhost:7860/
```

## Baseline inference (OpenAI API client)

Set API key (one-time in `.env`, auto-loaded each run):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key"
```

Run baseline script:

```bash
python scripts/run_baseline.py --model gpt-4o-mini
```

Or via endpoint:

```bash
curl -X POST http://localhost:7860/baseline -H "Content-Type: application/json" -d '{"model":"gpt-4o-mini"}'
```

### Reproducibility notes

- fixed task order
- deterministic environment transitions and grader
- model generation with `temperature=0`, `top_p=1`

## Baseline scores

Reproducible reference run from this repo (deterministic fallback mode, no API key):

```json
{
  "model": "gpt-4o-mini",
  "average_score": 0.4356,
  "task_results": [
    {"task_id": "support-easy-001", "score": 1.0, "steps": 8, "done": true},
    {"task_id": "support-medium-001", "score": 0.2111, "steps": 22, "done": true},
    {"task_id": "support-hard-001", "score": 0.0958, "steps": 30, "done": true}
  ]
}
```

When `OPENAI_API_KEY` is set, the same script/endpoints run through the OpenAI API client (`temperature=0`) and can be used as your model baseline.

## Docker

Build and run:

```bash
docker build -t openenv-support-triage .
docker run --rm -p 7860:7860 openenv-support-triage
```

## One-command pre-submission check

Run this from PowerShell:

```bash
./precheck.ps1
```

Run this from Linux/macOS shell:

```bash
chmod +x precheck.sh
./precheck.sh
```

What it validates automatically:

- `openenv validate` (when `openenv` CLI is installed)
- Docker build succeeds
- Container starts on port 7860
- `/`, `/reset`, `/tasks`, `/grader`, `/baseline` respond
- 3+ task scores returned and all scores are in `[0.0, 1.0]`

## CI automation

GitHub Actions workflow is included at [ .github/workflows/precheck.yml ](.github/workflows/precheck.yml).

It runs `./precheck.sh` on every push and pull request, so Docker/build/endpoint regressions are caught before submission.

## Submission guide

Use [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for the final step-by-step submission gate.

## Hugging Face Spaces deployment (Docker)

1. Create a new **Docker Space**.
2. Push this repository contents.
3. Ensure Space metadata includes the `openenv` tag (already in this README frontmatter).
4. Optionally set `OPENAI_API_KEY` in Space secrets to enable `/baseline`.

The app serves on port `7860` and responds to `POST /reset` for validator checks.

## Validation checklist mapping

- ✅ Real-world domain (support operations triage)
- ✅ 3+ tasks with deterministic graders (easy/medium/hard)
- ✅ Typed models and `step/reset/state`
- ✅ Dense reward shaping with penalties and partial progress
- ✅ Baseline script using OpenAI API client + env var key
- ✅ Dockerfile for containerized execution
- ✅ HF Space Docker-compatible metadata and runtime endpoints
