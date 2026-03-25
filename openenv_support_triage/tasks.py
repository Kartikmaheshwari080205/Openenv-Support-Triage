from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .models import Difficulty


class TicketSpec(BaseModel):
    ticket_id: str
    customer: str
    subject: str
    body: str
    priority: int = Field(ge=1, le=5)
    sla_hours: int = Field(gt=0)
    true_category: str
    true_team: str
    required_keywords: List[str]
    forbidden_keywords: List[str] = []


class TaskSpec(BaseModel):
    task_id: str
    name: str
    difficulty: Difficulty
    description: str
    objective: str
    max_steps: int
    guidance: str
    tickets: List[TicketSpec]


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="support-easy-001",
        name="Billing + Login Triage",
        difficulty=Difficulty.EASY,
        description="Triages two straightforward support tickets with clear routing.",
        objective="Classify, assign, draft a helpful reply, and close both tickets.",
        max_steps=14,
        guidance="Prioritize correctness over speed. Each ticket should be classified, assigned, replied to, then closed.",
        tickets=[
            TicketSpec(
                ticket_id="E1",
                customer="Ava Perez",
                subject="Charged twice for monthly plan",
                body="I upgraded once but got two charges on my card. Please fix and tell me when I get my money back.",
                priority=3,
                sla_hours=24,
                true_category="billing",
                true_team="billing",
                required_keywords=["refund", "double charge", "timeline"],
                forbidden_keywords=["cannot help"],
            ),
            TicketSpec(
                ticket_id="E2",
                customer="Noah Singh",
                subject="Cannot log in after password reset",
                body="Password reset says successful, but login still fails with invalid credentials.",
                priority=3,
                sla_hours=24,
                true_category="technical",
                true_team="technical",
                required_keywords=["troubleshoot", "reset link", "support"],
                forbidden_keywords=["ignore"],
            ),
        ],
    ),
    TaskSpec(
        task_id="support-medium-001",
        name="Priority SLA Handling",
        difficulty=Difficulty.MEDIUM,
        description="Manages mixed urgency inbox with VIP and logistics constraints.",
        objective="Meet SLA intent while maintaining accurate routing and policy-safe replies.",
        max_steps=22,
        guidance="High-priority and VIP tickets should be handled first; avoid overpromising outcomes.",
        tickets=[
            TicketSpec(
                ticket_id="M1",
                customer="Liam Chen (VIP)",
                subject="Enterprise invoice mismatch",
                body="Our invoice has 14 seats but we only use 11. Need corrected invoice today for procurement.",
                priority=5,
                sla_hours=4,
                true_category="billing",
                true_team="billing",
                required_keywords=["vip", "invoice", "same-day"],
                forbidden_keywords=["48 hours"],
            ),
            TicketSpec(
                ticket_id="M2",
                customer="Mia Johnson",
                subject="Package not delivered",
                body="Tracking has been stuck for three days. This was a birthday gift.",
                priority=4,
                sla_hours=8,
                true_category="logistics",
                true_team="logistics",
                required_keywords=["carrier", "investigate", "update"],
                forbidden_keywords=["not our problem"],
            ),
            TicketSpec(
                ticket_id="M3",
                customer="Ethan Müller",
                subject="Account locked after unusual login alert",
                body="My account is locked and I cannot access my orders dashboard.",
                priority=4,
                sla_hours=8,
                true_category="security",
                true_team="security",
                required_keywords=["identity", "secure", "unlock"],
                forbidden_keywords=["share password"],
            ),
        ],
    ),
    TaskSpec(
        task_id="support-hard-001",
        name="Fraud, Policy, and Escalation",
        difficulty=Difficulty.HARD,
        description="Requires nuanced handling of potentially fraudulent orders, legal-sensitive refund language, and multi-team routing.",
        objective="Resolve all tickets safely with correct escalation and policy-compliant communication.",
        max_steps=30,
        guidance="Security-sensitive tickets must be escalated correctly; never promise guaranteed outcomes on investigations.",
        tickets=[
            TicketSpec(
                ticket_id="H1",
                customer="Olivia García",
                subject="Unauthorized purchase on account",
                body="I see a purchase I did not make. Freeze account and reverse transaction now.",
                priority=5,
                sla_hours=2,
                true_category="fraud",
                true_team="security",
                required_keywords=["fraud", "investigation", "freeze"],
                forbidden_keywords=["guaranteed refund"],
            ),
            TicketSpec(
                ticket_id="H2",
                customer="Lucas Brown",
                subject="Refund requested outside standard window",
                body="I am at day 44. Product failed and I need a refund exception.",
                priority=4,
                sla_hours=12,
                true_category="billing",
                true_team="billing",
                required_keywords=["policy", "review", "exception"],
                forbidden_keywords=["automatic refund"],
            ),
            TicketSpec(
                ticket_id="H3",
                customer="Sofia Rossi",
                subject="Repeated shipment to wrong address",
                body="Third replacement was sent to old address despite profile update.",
                priority=4,
                sla_hours=8,
                true_category="logistics",
                true_team="logistics",
                required_keywords=["address", "carrier", "escalate"],
                forbidden_keywords=["user error"],
            ),
            TicketSpec(
                ticket_id="H4",
                customer="Amir Haddad",
                subject="Potential data exposure in support thread",
                body="Another customer's details appeared in my email thread. Need immediate response.",
                priority=5,
                sla_hours=2,
                true_category="security",
                true_team="security",
                required_keywords=["incident", "security", "urgent"],
                forbidden_keywords=["minor issue"],
            ),
        ],
    ),
]


def get_task_spec(task_id: str) -> TaskSpec:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Unknown task_id: {task_id}")
