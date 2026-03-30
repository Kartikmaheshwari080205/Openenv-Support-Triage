from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openenv_support_triage.environment import SupportTriageEnv
from openenv_support_triage.models import Action, BaselineResult, EnvironmentState, StepResponse, TasksResponse


def _safe_load_dotenv() -> None:
    try:
        load_dotenv()
    except UnicodeDecodeError:
        # Ignore malformed local .env files so the API can still start.
        pass


_safe_load_dotenv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"


app = FastAPI(title="OpenEnv Support Triage", version="1.0.0")
env = SupportTriageEnv()


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "service": "openenv-support-triage",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None):
    task_id = payload.task_id if payload else None
    try:
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    return env.step(action)


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    return env.state()


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    return env.tasks()


@app.get("/grader")
def grader() -> dict:
    current_state = env.state()
    return {
        "task_id": current_state.task_id,
        "done": current_state.done,
        "score": env.grade(),
        "step_count": current_state.step_count,
    }


@app.post("/baseline", response_model=BaselineResult)
def baseline(payload: Optional[BaselineRequest] = None) -> BaselineResult:
    model = payload.model if payload else "gpt-4o-mini"
    try:
        from openenv_support_triage.baseline import run_baseline

        return run_baseline(model=model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
