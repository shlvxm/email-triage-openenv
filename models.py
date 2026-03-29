from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import IntEnum

class EmailAction(IntEnum):
    SPAM = 0
    PRIMARY = 1
    SOCIAL = 2
    PROMO = 3
    URGENT = 4
    REPLY = 5

class EmailObservation(BaseModel):
    features: List[float] = Field(..., description="10-dimensional feature vector [0, 1]")
    summary: str = Field(..., description="Natural language summary of the email")
    feature_names: List[str] = Field(default=["urgency", "keyword", "sender", "spam", "promo", "social", "reply", "attach", "tod", "thread"])

class EmailState(BaseModel):
    step_count: int
    episode_id: str
    max_steps: int
    true_label: int
    total_reward: float

class Reward(BaseModel):
    value: float = Field(..., description="Floating point reward [0.0, 1.0]")
    is_correct: bool = Field(..., description="Whether the action matched the true label")
    reason: Optional[str] = Field(None, description="Explanation for the reward value")

class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict
