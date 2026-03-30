from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from .environment import SupportTriageEnv
from .models import Action, BaselineResult, BaselineTaskResult
from .tasks import TASKS


def _safe_load_dotenv() -> None:
    try:
        load_dotenv()
    except UnicodeDecodeError:
        # Ignore malformed local .env files and rely on process env vars.
        pass


def _resolve_client_config() -> Tuple[Optional[str], Optional[str], str]:
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return api_base_url, api_key, model_name


def _model_action(client: OpenAI, model: str, observation_json: str) -> Action:
    system_prompt = (
        "You are a customer support triage agent. Return ONLY JSON with fields: "
        "action_type, ticket_id, value, message. "
        "Choose one action at a time to maximize final grader score."
    )

    user_prompt = (
        "Observation:\n"
        f"{observation_json}\n\n"
        "Rules:\n"
        "1) classify each ticket correctly.\n"
        "2) assign to the right team.\n"
        "3) write a concise helpful reply containing relevant policy keywords.\n"
        "4) close only after classify+assign+reply are done.\n"
        "Return JSON only."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    payload: Dict[str, str] = json.loads(content)
    return Action.model_validate(payload)


def _infer_category(subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()
    if any(token in text for token in ["unauthorized", "fraud", "exposure", "locked", "security"]):
        return "security"
    if any(token in text for token in ["invoice", "refund", "charged", "billing", "payment"]):
        return "billing"
    if any(token in text for token in ["shipment", "package", "carrier", "delivered", "address"]):
        return "logistics"
    if any(token in text for token in ["login", "password", "reset", "error", "cannot log"]):
        return "technical"
    return "technical"


def _team_for_category(category: str) -> str:
    mapping = {
        "billing": "billing",
        "technical": "technical",
        "logistics": "logistics",
        "security": "security",
        "fraud": "security",
    }
    return mapping.get(category, "general")


def _deterministic_action(observation: Dict[str, object]) -> Action:
    tickets = observation.get("tickets", [])
    open_tickets = [ticket for ticket in tickets if not ticket.get("closed")]

    if not open_tickets:
        return Action(action_type="noop")

    ranked = sorted(open_tickets, key=lambda item: (-int(item.get("priority", 1)), int(item.get("sla_hours", 999))))
    ticket = ranked[0]
    ticket_id = str(ticket["ticket_id"])

    if not ticket.get("predicted_category"):
        category = _infer_category(str(ticket.get("subject", "")), str(ticket.get("body", "")))
        return Action(action_type="classify", ticket_id=ticket_id, value=category)

    if not ticket.get("assigned_team"):
        category = str(ticket.get("predicted_category", "technical"))
        return Action(action_type="assign", ticket_id=ticket_id, value=_team_for_category(category))

    if not ticket.get("reply_draft"):
        category = str(ticket.get("predicted_category", "technical"))
        templates = {
            "billing": "Thanks for reporting this. We are reviewing the invoice and refund policy, including timeline and any double charge.",
            "technical": "Thanks for reaching out. We will troubleshoot account access using a fresh reset link and continue support until resolved.",
            "logistics": "Thanks for the update. We will investigate with the carrier, verify the address, and share an update shortly.",
            "security": "Thanks for alerting us. We started a security investigation, will secure your account, and provide an urgent incident update.",
            "fraud": "We have initiated a fraud investigation, are freezing relevant activity, and will provide status updates promptly.",
        }
        return Action(action_type="draft_reply", ticket_id=ticket_id, message=templates.get(category, templates["technical"]))

    return Action(action_type="close", ticket_id=ticket_id)


def run_baseline(model: Optional[str] = None, max_steps_multiplier: float = 1.0) -> BaselineResult:
    _safe_load_dotenv()
    api_base_url, api_key, env_model_name = _resolve_client_config()
    resolved_model = model or env_model_name
    client: Optional[OpenAI] = None
    if api_key:
        client_kwargs = {"api_key": api_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = OpenAI(**client_kwargs)
    task_results: List[BaselineTaskResult] = []

    for task in TASKS:
        env = SupportTriageEnv(task_id=task.task_id)
        observation = env.reset(task_id=task.task_id)

        step_limit = int(task.max_steps * max_steps_multiplier)
        done = False

        while not done and env.state().step_count < step_limit:
            if client is not None:
                try:
                    action = _model_action(client=client, model=resolved_model, observation_json=observation.model_dump_json(indent=2))
                except Exception:
                    action = _deterministic_action(observation.model_dump())
            else:
                action = _deterministic_action(observation.model_dump())

            result = env.step(action)
            observation = result.observation
            done = result.done

        task_results.append(
            BaselineTaskResult(
                task_id=task.task_id,
                score=env.grade(),
                steps=env.state().step_count,
                done=env.state().done,
            )
        )

    average_score = round(sum(item.score for item in task_results) / len(task_results), 4)
    return BaselineResult(
        model=resolved_model,
        seed_note="Uses OpenAI client with API_BASE_URL + HF_TOKEN/OPENAI_API_KEY + MODEL_NAME when configured; otherwise deterministic fallback policy",
        average_score=average_score,
        task_results=task_results,
    )
