from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

from .models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    Reward,
    StepInfo,
    StepResponse,
    TaskSummary,
    TasksResponse,
    TicketObservation,
    TicketState,
)
from .tasks import TASKS, TaskSpec, get_task_spec


class SupportTriageEnv:
    def __init__(self, task_id: str = TASKS[0].task_id) -> None:
        self.current_task: TaskSpec = get_task_spec(task_id)
        self.state_data: EnvironmentState = self._build_initial_state(self.current_task)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self.current_task = get_task_spec(task_id)
        self.state_data = self._build_initial_state(self.current_task)
        return self._observation()

    def state(self) -> EnvironmentState:
        return deepcopy(self.state_data)

    def step(self, action: Action) -> StepResponse:
        if self.state_data.done:
            reward = Reward(total=-0.02, components={"episode_done": -0.02}, rationale="Episode already finished")
            info = StepInfo(
                valid_action=False,
                grader_score=self.grade(),
                message="Episode is already done. Call reset().",
                reward_breakdown=reward.components,
                action_trace=action.model_dump(),
            )
            return StepResponse(observation=self._observation(), reward=reward, done=True, info=info)

        self.state_data.step_count += 1
        reward_components: Dict[str, float] = {"step_cost": -0.01}
        valid_action = True
        message = "Action applied"

        if action.action_type == ActionType.NOOP:
            reward_components["noop_penalty"] = -0.03
        else:
            valid_action, action_reward, message = self._apply_action(action)
            reward_components.update(action_reward)

        trace = self._action_trace(action)
        repeat_count = self.state_data.recent_actions.count(trace)
        if repeat_count >= 2:
            reward_components["loop_penalty"] = -0.05

        self.state_data.recent_actions = (self.state_data.recent_actions + [trace])[-8:]

        if self.state_data.step_count >= self.state_data.max_steps:
            self.state_data.done = True
            reward_components["max_steps_reached"] = -0.1
            message = "Episode ended at max steps"

        if all(ticket.closed for ticket in self.state_data.tickets.values()):
            self.state_data.done = True
            reward_components["all_tickets_closed"] = 0.15
            message = "All tickets closed"

        reward_value = max(-1.0, min(1.0, round(sum(reward_components.values()), 4)))
        self.state_data.cumulative_reward += reward_value

        score = self.grade() if self.state_data.done else None
        reward = Reward(total=reward_value, components=reward_components, rationale=message)
        info = StepInfo(
            valid_action=valid_action,
            grader_score=score,
            message=message,
            reward_breakdown=reward_components,
            action_trace=action.model_dump(),
        )
        return StepResponse(observation=self._observation(), reward=reward, done=self.state_data.done, info=info)

    def grade(self) -> float:
        ticket_scores = []
        for ticket in self.state_data.tickets.values():
            classification = 1.0 if ticket.predicted_category == ticket.true_category else 0.0
            assignment = 1.0 if ticket.assigned_team == ticket.true_team else 0.0
            reply_quality = self._reply_quality(ticket)
            closure = 1.0 if (ticket.closed and classification == 1.0 and assignment == 1.0 and reply_quality >= 0.6) else 0.0

            ticket_score = 0.25 * classification + 0.25 * assignment + 0.25 * reply_quality + 0.25 * closure
            ticket_scores.append(ticket_score)

        if not ticket_scores:
            return 0.0

        task_score = sum(ticket_scores) / len(ticket_scores)
        return round(max(0.0, min(1.0, task_score)), 4)

    def tasks(self) -> TasksResponse:
        return TasksResponse(
            tasks=[
                TaskSummary(
                    task_id=task.task_id,
                    name=task.name,
                    difficulty=task.difficulty,
                    description=task.description,
                    objective=task.objective,
                )
                for task in TASKS
            ],
            action_schema={
                "action_type": "classify | assign | draft_reply | close | noop",
                "ticket_id": "required for classify/assign/draft_reply/close",
                "value": "required for classify/assign",
                "message": "required for draft_reply",
                "categories": ["billing", "technical", "logistics", "security", "fraud"],
                "teams": ["billing", "technical", "logistics", "security", "general"],
            },
        )

    def _build_initial_state(self, task: TaskSpec) -> EnvironmentState:
        tickets = {
            item.ticket_id: TicketState(
                ticket_id=item.ticket_id,
                customer=item.customer,
                subject=item.subject,
                body=item.body,
                priority=item.priority,
                sla_hours=item.sla_hours,
                true_category=item.true_category,
                true_team=item.true_team,
                required_keywords=item.required_keywords,
                forbidden_keywords=item.forbidden_keywords,
            )
            for item in task.tickets
        }

        return EnvironmentState(
            task_id=task.task_id,
            task_name=task.name,
            difficulty=task.difficulty,
            step_count=0,
            max_steps=task.max_steps,
            done=False,
            tickets=tickets,
            recent_actions=[],
            cumulative_reward=0.0,
        )

    def _observation(self) -> Observation:
        return Observation(
            task_id=self.state_data.task_id,
            task_name=self.state_data.task_name,
            difficulty=self.state_data.difficulty,
            step_count=self.state_data.step_count,
            max_steps=self.state_data.max_steps,
            tickets=[
                TicketObservation(
                    ticket_id=t.ticket_id,
                    customer=t.customer,
                    subject=t.subject,
                    body=t.body,
                    priority=t.priority,
                    sla_hours=t.sla_hours,
                    predicted_category=t.predicted_category,
                    assigned_team=t.assigned_team,
                    reply_draft=t.reply_draft,
                    closed=t.closed,
                )
                for t in self.state_data.tickets.values()
            ],
            recent_actions=self.state_data.recent_actions,
            guidance=self.current_task.guidance,
        )

    def _apply_action(self, action: Action) -> Tuple[bool, Dict[str, float], str]:
        rewards: Dict[str, float] = {}
        ticket = self.state_data.tickets.get(action.ticket_id or "")

        if ticket is None:
            return False, {"invalid_action": -0.1}, "Unknown or missing ticket_id"

        if action.action_type == ActionType.CLASSIFY:
            if not action.value:
                return False, {"invalid_action": -0.1}, "Missing category value"
            if ticket.closed:
                return False, {"destructive_penalty": -0.12}, "Cannot classify a closed ticket"

            if ticket.predicted_category is not None:
                rewards["reclassification_penalty"] = -0.02

            ticket.predicted_category = action.value.strip().lower()
            if ticket.predicted_category == ticket.true_category:
                rewards["classification_correct"] = 0.2
                return True, rewards, "Category classified correctly"
            rewards["classification_incorrect"] = -0.05
            return True, rewards, "Category classified incorrectly"

        if action.action_type == ActionType.ASSIGN:
            if not action.value:
                return False, {"invalid_action": -0.1}, "Missing team value"
            if ticket.closed:
                return False, {"destructive_penalty": -0.12}, "Cannot assign a closed ticket"

            if ticket.assigned_team is not None:
                rewards["reassignment_penalty"] = -0.02

            ticket.assigned_team = action.value.strip().lower()
            if ticket.assigned_team == ticket.true_team:
                rewards["assignment_correct"] = 0.2
                return True, rewards, "Ticket assigned correctly"
            rewards["assignment_incorrect"] = -0.05
            return True, rewards, "Ticket assigned incorrectly"

        if action.action_type == ActionType.DRAFT_REPLY:
            if not action.message:
                return False, {"invalid_action": -0.1}, "Missing reply message"
            if ticket.closed:
                return False, {"destructive_penalty": -0.12}, "Cannot draft reply for a closed ticket"

            old_quality = self._reply_quality(ticket)
            ticket.reply_draft = action.message
            new_quality = self._reply_quality(ticket)
            improvement = round((new_quality - old_quality) * 0.25, 4)
            rewards["reply_quality_progress"] = improvement
            if new_quality >= 0.8:
                rewards["high_quality_reply"] = 0.08
            return True, rewards, "Reply updated"

        if action.action_type == ActionType.CLOSE:
            if ticket.closed:
                return False, {"redundant_close_penalty": -0.05}, "Ticket already closed"

            is_classified = ticket.predicted_category == ticket.true_category
            is_assigned = ticket.assigned_team == ticket.true_team
            quality = self._reply_quality(ticket)

            if is_classified and is_assigned and quality >= 0.6:
                ticket.closed = True
                rewards["good_close"] = 0.3
                return True, rewards, "Ticket closed with sufficient quality"

            rewards["premature_close_penalty"] = -0.2
            return True, rewards, "Ticket closed prematurely"

        return False, {"invalid_action": -0.1}, "Unsupported action_type"

    def _reply_quality(self, ticket: TicketState) -> float:
        if not ticket.reply_draft:
            return 0.0

        draft = ticket.reply_draft.lower()

        required_hits = sum(1 for keyword in ticket.required_keywords if keyword.lower() in draft)
        required_ratio = required_hits / max(1, len(ticket.required_keywords))

        forbidden_hits = sum(1 for phrase in ticket.forbidden_keywords if phrase.lower() in draft)
        forbidden_score = 1.0 if forbidden_hits == 0 else max(0.0, 1.0 - (forbidden_hits / max(1, len(ticket.forbidden_keywords))))

        quality = 0.7 * required_ratio + 0.3 * forbidden_score
        return round(max(0.0, min(1.0, quality)), 4)

    @staticmethod
    def _action_trace(action: Action) -> str:
        return "|".join(
            [
                action.action_type.value,
                action.ticket_id or "",
                (action.value or "").strip().lower(),
                ((action.message or "")[:64]).strip().lower(),
            ]
        )
