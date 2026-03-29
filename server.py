import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from env import EmailTriageEnv
from models import EmailAction, EmailObservation, EmailState, StepResult

app = FastAPI(title="Email Triage OpenEnv API", description="OpenEnv-compliant server for Email Triage RL")

# Global env instance for simplicity (suitable for single-agent evaluation bots)
current_env: Optional[EmailTriageEnv] = None

class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    task_id: Optional[str] = "hard"

@app.post("/reset", response_model=EmailObservation)
def reset_env(request: Optional[ResetRequest] = None):
    """Initializes a new episode and returns the initial observation."""
    global current_env
    task_level = request.task_id if request else "hard"
    current_env = EmailTriageEnv(task_level=task_level)
    obs = current_env.reset()
    return obs

@app.post("/step", response_model=StepResult)
def step_env(request: StepRequest):
    """Executes an action within the environment."""
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        action_enum = EmailAction(request.action)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
    try:
        step_result = current_env.step(action_enum)
        return step_result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=EmailState)
def get_state():
    """Retrieves current episode metadata."""
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return current_env.state()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "openenv-email-triage"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
