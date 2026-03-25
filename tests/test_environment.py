from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.models import Action, ActionType


def test_reset_returns_initial_observation() -> None:
    env = SupportTriageEnv()
    obs = env.reset()
    assert obs.step_count == 0
    assert len(obs.tickets) >= 2


def test_step_updates_state_and_returns_reward() -> None:
    env = SupportTriageEnv(task_id="support-easy-001")
    env.reset(task_id="support-easy-001")

    ticket_id = env.state().tickets["E1"].ticket_id
    response = env.step(Action(action_type=ActionType.CLASSIFY, ticket_id=ticket_id, value="billing"))

    assert response.reward.total != 0
    assert response.info.valid_action is True
    assert env.state().step_count == 1


def test_grader_range() -> None:
    env = SupportTriageEnv(task_id="support-hard-001")
    env.reset(task_id="support-hard-001")
    score = env.grade()
    assert 0.0 <= score <= 1.0
