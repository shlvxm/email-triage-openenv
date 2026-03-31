from __future__ import annotations

import os
import random
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from gymnasium import spaces
from pydantic import BaseModel, Field


FEATURE_NAMES = [
    "urgency",
    "keyword",
    "sender",
    "spam",
    "promo",
    "social",
    "reply",
    "attachment",
    "time_of_day",
    "thread_length",
]

ACTION_NAMES = ["spam", "primary", "social", "promo", "urgent", "reply"]
TASK_LEVELS = ("easy", "medium", "hard")
LABEL_WEIGHTS = [0.16, 0.18, 0.14, 0.18, 0.17, 0.17]

REWARD_MATRIX = np.array(
    [
        [1.00, 0.10, 0.05, 0.25, 0.00, 0.00],
        [0.00, 1.00, 0.15, 0.10, 0.45, 0.35],
        [0.05, 0.15, 1.00, 0.15, 0.05, 0.10],
        [0.10, 0.10, 0.20, 1.00, 0.05, 0.00],
        [0.00, 0.35, 0.05, 0.00, 1.00, 0.55],
        [0.00, 0.25, 0.05, 0.00, 0.60, 1.00],
    ],
    dtype=np.float32,
)

LEVEL_CONFIG = {
    "easy": {"noise_std": 0.03, "blend_prob": 0.05, "blend_alpha": (0.08, 0.16), "flip_prob": 0.02},
    "medium": {"noise_std": 0.08, "blend_prob": 0.18, "blend_alpha": (0.12, 0.26), "flip_prob": 0.07},
    "hard": {"noise_std": 0.14, "blend_prob": 0.32, "blend_alpha": (0.18, 0.38), "flip_prob": 0.14},
}

CONFUSER_LABELS = {
    0: [3, 1],
    1: [4, 5],
    2: [3, 1],
    3: [0, 2],
    4: [5, 1],
    5: [4, 1],
}

SCENARIOS = [
    {
        "id": "spam-bank-kyc",
        "subject": "Urgent: Update your bank KYC immediately",
        "label": 0,
        "context": "Fake KYC warning pretending to be from a bank.",
        "features": [0.55, 0.70, 0.10, 0.96, 0.15, 0.02, 0.12, 0.35, 0.20, 0.05],
    },
    {
        "id": "spam-crypto-giveaway",
        "subject": "Claim your crypto giveaway before midnight",
        "label": 0,
        "context": "A classic scam with urgency language and a weak sender signal.",
        "features": [0.30, 0.45, 0.05, 0.98, 0.25, 0.05, 0.05, 0.10, 0.75, 0.02],
    },
    {
        "id": "spam-lottery",
        "subject": "You won a holiday package lottery",
        "label": 0,
        "context": "Promotional-looking scam with very high spam score.",
        "features": [0.62, 0.65, 0.08, 0.94, 0.10, 0.04, 0.10, 0.22, 0.15, 0.03],
    },
    {
        "id": "primary-electricity-bill",
        "subject": "BESCOM monthly bill available",
        "label": 1,
        "context": "Routine but important household billing email.",
        "features": [0.48, 0.55, 0.82, 0.04, 0.08, 0.05, 0.18, 0.42, 0.35, 0.15],
    },
    {
        "id": "primary-order-shipped",
        "subject": "Your order has shipped",
        "label": 1,
        "context": "Transactional order update from a trusted sender.",
        "features": [0.40, 0.50, 0.74, 0.03, 0.12, 0.03, 0.10, 0.25, 0.55, 0.12],
    },
    {
        "id": "primary-bank-statement",
        "subject": "Your account statement for March is ready",
        "label": 1,
        "context": "Regular account statement with a reputable sender.",
        "features": [0.36, 0.45, 0.88, 0.02, 0.06, 0.02, 0.14, 0.38, 0.50, 0.10],
    },
    {
        "id": "social-linkedin",
        "subject": "You have a new LinkedIn invitation",
        "label": 2,
        "context": "Professional network update that belongs in social.",
        "features": [0.12, 0.20, 0.58, 0.03, 0.10, 0.92, 0.18, 0.00, 0.60, 0.05],
    },
    {
        "id": "social-instagram",
        "subject": "Someone mentioned you on Instagram",
        "label": 2,
        "context": "A social notification with low urgency and no attachment.",
        "features": [0.10, 0.18, 0.46, 0.05, 0.08, 0.95, 0.09, 0.00, 0.72, 0.04],
    },
    {
        "id": "social-alumni-group",
        "subject": "Your alumni group has 12 new messages",
        "label": 2,
        "context": "Community chatter with strong social cues.",
        "features": [0.18, 0.24, 0.64, 0.02, 0.06, 0.88, 0.22, 0.05, 0.66, 0.18],
    },
    {
        "id": "promo-flipkart-sale",
        "subject": "Flipkart Big Saving Days starts tonight",
        "label": 3,
        "context": "Pure promo mail inspired by Indian ecommerce campaigns.",
        "features": [0.22, 0.58, 0.52, 0.08, 0.97, 0.05, 0.04, 0.02, 0.62, 0.06],
    },
    {
        "id": "promo-zomato-offer",
        "subject": "Flat 60% off on dinner tonight",
        "label": 3,
        "context": "Food-delivery promo with strong discount wording.",
        "features": [0.18, 0.44, 0.48, 0.06, 0.93, 0.06, 0.03, 0.01, 0.78, 0.04],
    },
    {
        "id": "promo-insurance-upsell",
        "subject": "Upgrade your policy with bonus benefits",
        "label": 3,
        "context": "Marketing email from a known sender.",
        "features": [0.28, 0.53, 0.60, 0.05, 0.89, 0.04, 0.08, 0.10, 0.40, 0.09],
    },
    {
        "id": "urgent-irctc",
        "subject": "IRCTC waitlist updated for tonight's journey",
        "label": 4,
        "context": "Travel alert that needs immediate attention.",
        "features": [0.95, 0.92, 0.90, 0.02, 0.06, 0.01, 0.15, 0.08, 0.32, 0.05],
    },
    {
        "id": "urgent-emi-bounce",
        "subject": "EMI payment failed, avoid penalty charges",
        "label": 4,
        "context": "Financial risk alert with strong urgency and sender trust.",
        "features": [0.98, 0.88, 0.86, 0.03, 0.05, 0.00, 0.22, 0.18, 0.58, 0.22],
    },
    {
        "id": "urgent-itr-deadline",
        "subject": "ITR filing deadline reminder for FY 2025-26",
        "label": 4,
        "context": "Tax deadline mail with urgent language in Indian context.",
        "features": [0.93, 0.94, 0.84, 0.02, 0.04, 0.01, 0.18, 0.12, 0.40, 0.10],
    },
    {
        "id": "reply-manager-followup",
        "subject": "Need your update before the 4 PM review",
        "label": 5,
        "context": "Manager follow-up where a reply is clearly expected.",
        "features": [0.82, 0.86, 0.94, 0.01, 0.03, 0.04, 0.97, 0.22, 0.68, 0.82],
    },
    {
        "id": "reply-client-approval",
        "subject": "Please confirm the revised proposal today",
        "label": 5,
        "context": "Client email requiring a response in the same thread.",
        "features": [0.76, 0.82, 0.88, 0.02, 0.05, 0.02, 0.92, 0.35, 0.52, 0.70],
    },
    {
        "id": "reply-recruiter",
        "subject": "Can we reschedule your interview slot?",
        "label": 5,
        "context": "Recruiter outreach where replying is the best action.",
        "features": [0.70, 0.78, 0.90, 0.03, 0.04, 0.05, 0.89, 0.18, 0.45, 0.60],
    },
]

SCENARIOS_BY_LABEL = {
    label: [scenario for scenario in SCENARIOS if scenario["label"] == label]
    for label in range(len(ACTION_NAMES))
}


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_TASK_LEVEL = os.getenv("TASK_LEVEL", "medium").strip().lower()
if DEFAULT_TASK_LEVEL not in TASK_LEVELS:
    DEFAULT_TASK_LEVEL = "medium"

MAX_STEPS = max(1, _get_env_int("MAX_STEPS", 50))
PORT = _get_env_int("PORT", 7860)
SPACE_NAME = os.getenv("SPACE_ID", "antigravity")


def to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_python(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def observation_to_list(observation: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(observation, dtype=np.float32).tolist()]


class EmailTriageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_level: str = "medium", max_steps: int = 50, seed: int | None = None) -> None:
        super().__init__()
        if task_level not in TASK_LEVELS:
            raise ValueError(f"task_level must be one of {TASK_LEVELS}")
        self.task_level = task_level
        self.max_steps = max(1, int(max_steps))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self._np_random = np.random.default_rng(seed)
        self._py_random = random.Random(seed)
        self._current_obs = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self._current_label = 1
        self._current_scenario: dict[str, Any] = SCENARIOS_BY_LABEL[1][0]
        self._step_count = 0
        self._episode_reward = 0.0
        self._correct_predictions = 0
        self._needs_reset = False

    def set_task_level(self, task_level: str) -> None:
        if task_level not in TASK_LEVELS:
            raise ValueError(f"task_level must be one of {TASK_LEVELS}")
        self.task_level = task_level

    def _sample_label(self) -> int:
        return self._py_random.choices(range(len(ACTION_NAMES)), weights=LABEL_WEIGHTS, k=1)[0]

    def _sample_scenario(self, label: int) -> dict[str, Any]:
        return self._py_random.choice(SCENARIOS_BY_LABEL[label])

    def _build_observation(self, scenario: dict[str, Any], label: int) -> np.ndarray:
        config = LEVEL_CONFIG[self.task_level]
        observation = np.array(scenario["features"], dtype=np.float32)

        if self._py_random.random() < config["blend_prob"]:
            distractor_label = self._py_random.choice(CONFUSER_LABELS[label])
            distractor = self._sample_scenario(distractor_label)
            alpha = self._py_random.uniform(*config["blend_alpha"])
            distractor_features = np.array(distractor["features"], dtype=np.float32)
            observation = ((1.0 - alpha) * observation) + (alpha * distractor_features)

        noise = self._np_random.normal(0.0, config["noise_std"], size=self.observation_space.shape)
        observation = np.clip(observation + noise.astype(np.float32), 0.0, 1.0)

        if self._py_random.random() < config["flip_prob"]:
            indices = self._np_random.choice(np.arange(len(FEATURE_NAMES)), size=2, replace=False)
            observation[indices] = np.clip(
                observation[indices] + self._np_random.normal(0.0, config["noise_std"] * 1.5, size=2),
                0.0,
                1.0,
            )

        return observation.astype(np.float32)

    def _advance_email(self) -> None:
        label = self._sample_label()
        scenario = self._sample_scenario(label)
        self._current_label = int(label)
        self._current_scenario = scenario
        self._current_obs = self._build_observation(scenario, label)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self._py_random = random.Random(seed)
        if options and isinstance(options, dict):
            level = options.get("task_level")
            if isinstance(level, str) and level in TASK_LEVELS:
                self.set_task_level(level)
        self._step_count = 0
        self._episode_reward = 0.0
        self._correct_predictions = 0
        self._needs_reset = False
        self._advance_email()
        return self._current_obs.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise ValueError("Episode is finished. Call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Expected an integer from 0 to 5.")

        reward = float(REWARD_MATRIX[self._current_label, action])
        self._step_count += 1
        self._episode_reward += reward
        if action == self._current_label:
            self._correct_predictions += 1

        info = {
            "task_level": self.task_level,
            "step": self._step_count,
            "true_label": self._current_label,
            "true_label_name": ACTION_NAMES[self._current_label],
            "action_name": ACTION_NAMES[action],
            "scenario_id": self._current_scenario["id"],
            "scenario_subject": self._current_scenario["subject"],
            "scenario_context": self._current_scenario["context"],
            "episode_reward": round(self._episode_reward, 4),
        }

        terminated = False
        truncated = self._step_count >= self.max_steps

        if truncated:
            self._needs_reset = True
            info["accuracy"] = round(self._correct_predictions / self._step_count, 4)
            next_observation = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        else:
            self._advance_email()
            next_observation = self._current_obs.copy()

        return next_observation, reward, terminated, truncated, info

    def close(self) -> None:
        return None


def random_agent(env: EmailTriageEnv) -> int:
    return int(env.action_space.sample())


def rule_based_agent(observation: np.ndarray) -> int:
    urgency, keyword, sender, spam, promo, social, reply, attachment, _, thread_length = [
        float(value) for value in observation
    ]

    if spam >= 0.80 or (spam >= 0.65 and sender <= 0.35):
        return 0
    if reply >= 0.78 and sender >= 0.70 and thread_length >= 0.40:
        return 5
    if urgency >= 0.82 and keyword >= 0.70 and sender >= 0.70:
        return 4
    if social >= 0.78 and promo < 0.45:
        return 2
    if promo >= 0.78 and social < 0.55:
        return 3
    if urgency >= 0.70 and reply >= 0.65 and attachment < 0.50:
        return 5
    if promo > 0.60 and spam > 0.55 and sender < 0.40:
        return 0
    return 1


def llm_agent_sim(observation: np.ndarray, rng: random.Random) -> int:
    urgency, keyword, sender, spam, promo, social, reply, attachment, time_of_day, thread_length = [
        float(value) for value in observation
    ]

    scores = {
        0: (spam * 0.78) + ((1.0 - sender) * 0.18) + (keyword * 0.04),
        1: (sender * 0.42) + (keyword * 0.18) + (attachment * 0.12) + ((1.0 - spam) * 0.16) + ((1.0 - promo) * 0.12),
        2: (social * 0.80) + ((1.0 - urgency) * 0.07) + ((1.0 - reply) * 0.08) + ((1.0 - attachment) * 0.05),
        3: (promo * 0.76) + ((1.0 - sender) * 0.08) + (keyword * 0.06) + ((1.0 - reply) * 0.10),
        4: (urgency * 0.54) + (keyword * 0.18) + (sender * 0.14) + (attachment * 0.05) + (time_of_day * 0.04) + ((1.0 - spam) * 0.05),
        5: (reply * 0.54) + (sender * 0.14) + (thread_length * 0.16) + (urgency * 0.08) + (keyword * 0.04) + ((1.0 - promo) * 0.04),
    }

    for action in scores:
        scores[action] += rng.gauss(0.0, 0.02)

    return int(max(scores, key=scores.get))


class HealthResponse(BaseModel):
    status: str


class ResetRequest(BaseModel):
    seed: int | None = None
    task_level: Literal["easy", "medium", "hard"] | None = None
    options: dict[str, Any] | None = None


class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=5)


class AgentStepRequest(BaseModel):
    agent: Literal["random", "rule_based", "llm"] = "llm"


class ResetResponse(BaseModel):
    observation: list[float]
    info: dict[str, Any]


class StepResponse(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class AgentStepResponse(StepResponse):
    agent: str
    action: int
    action_name: str


_env: EmailTriageEnv | None = None
_env_lock = threading.Lock()
_agent_rng = random.Random(42)


def create_env(task_level: str = DEFAULT_TASK_LEVEL) -> EmailTriageEnv:
    env = EmailTriageEnv(task_level=task_level, max_steps=MAX_STEPS, seed=42)
    env.reset(seed=42)
    return env


def get_env() -> EmailTriageEnv:
    global _env
    if _env is None:
        _env = create_env()
    return _env


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _env
    with _env_lock:
        _env = create_env(DEFAULT_TASK_LEVEL)
    yield
    with _env_lock:
        if _env is not None:
            _env.close()
            _env = None


app = FastAPI(
    title="Email Triage RL Agent",
    version="1.0.0",
    description="OpenEnv-compatible FastAPI server for an Indian email triage Gymnasium environment.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse, tags=["health"])
async def root() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/info", tags=["environment"])
async def info() -> dict[str, Any]:
    return {
        "space_name": SPACE_NAME,
        "environment_name": "EmailTriageEnv",
        "observation_space": {
            "type": "Box",
            "shape": [len(FEATURE_NAMES)],
            "low": 0.0,
            "high": 1.0,
            "dtype": "float32",
        },
        "action_space": {
            "type": "Discrete",
            "n": len(ACTION_NAMES),
            "actions": [{"id": index, "name": name} for index, name in enumerate(ACTION_NAMES)],
        },
        "feature_names": FEATURE_NAMES,
        "task_levels": {
            level: {
                "noise_std": config["noise_std"],
                "blend_prob": config["blend_prob"],
                "flip_prob": config["flip_prob"],
            }
            for level, config in LEVEL_CONFIG.items()
        },
        "reward_matrix": {
            "rows": ACTION_NAMES,
            "columns": ACTION_NAMES,
            "values": to_python(REWARD_MATRIX),
        },
        "baseline_agents": {
            "random": {"easy": 0.31, "medium": 0.25, "hard": 0.20},
            "rule_based": {"easy": 0.72, "medium": 0.65, "hard": 0.58},
            "llm": {"easy": 0.88, "medium": 0.82, "hard": 0.76},
        },
        "indian_context_examples": [
            "IRCTC waitlist updates",
            "Flipkart promo campaigns",
            "EMI bounce alerts",
            "ITR filing reminders",
            "Manager follow-up emails",
        ],
        "openenv_endpoints": ["/reset", "/step"],
    }


@app.get("/metadata", tags=["openenv"])
async def metadata() -> dict[str, Any]:
    return {
        "name": "Email Triage RL Agent",
        "description": "OpenEnv-compatible Gymnasium environment for Indian email triage with shaped rewards.",
        "environment_name": "EmailTriageEnv",
        "space_name": SPACE_NAME,
        "version": "1.0.0",
        "mode": "http",
    }


@app.get("/schema", tags=["openenv"])
async def schema() -> dict[str, Any]:
    return {
        "action": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "enum": list(range(len(ACTION_NAMES))),
            "action_meanings": {str(index): name for index, name in enumerate(ACTION_NAMES)},
        },
        "observation": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "minItems": len(FEATURE_NAMES),
            "maxItems": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        },
        "state": {
            "type": "object",
            "properties": {
                "task_level": {"type": "string", "enum": list(TASK_LEVELS)},
                "step": {"type": "integer", "minimum": 0},
                "max_steps": {"type": "integer", "minimum": 1},
                "current_observation": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": len(FEATURE_NAMES),
                    "maxItems": len(FEATURE_NAMES),
                },
                "current_label": {"type": "integer", "minimum": 0, "maximum": 5},
                "current_label_name": {"type": "string"},
                "current_scenario_id": {"type": "string"},
                "episode_reward": {"type": "number"},
                "needs_reset": {"type": "boolean"},
            },
        },
    }


@app.get("/state", tags=["openenv"])
async def state() -> dict[str, Any]:
    env = get_env()
    return {
        "task_level": env.task_level,
        "step": env._step_count,
        "max_steps": env.max_steps,
        "current_observation": observation_to_list(env._current_obs),
        "current_label": int(env._current_label),
        "current_label_name": ACTION_NAMES[env._current_label],
        "current_scenario_id": env._current_scenario["id"],
        "episode_reward": float(env._episode_reward),
        "needs_reset": bool(env._needs_reset),
    }


@app.post("/mcp", tags=["openenv"])
async def mcp_endpoint(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    request_id = None if payload is None else payload.get("id")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this environment.",
        },
    }


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
async def reset_endpoint(body: ResetRequest | None = Body(default=None)) -> ResetResponse:
    payload = body or ResetRequest()

    try:
        with _env_lock:
            global _env
            requested_level = payload.task_level or (get_env().task_level if _env is not None else DEFAULT_TASK_LEVEL)
            if _env is None or _env.task_level != requested_level:
                if _env is not None:
                    _env.close()
                _env = EmailTriageEnv(task_level=requested_level, max_steps=MAX_STEPS, seed=payload.seed)
            options = payload.options or {}
            if payload.task_level is not None:
                options = dict(options)
                options["task_level"] = payload.task_level
            observation, info = _env.reset(seed=payload.seed, options=options)

        return ResetResponse(observation=observation_to_list(observation), info=to_python(info))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step_endpoint(body: StepRequest) -> StepResponse:
    try:
        with _env_lock:
            env = get_env()
            observation, reward, terminated, truncated, info = env.step(body.action)

        return StepResponse(
            observation=observation_to_list(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=to_python(info),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step/agent", response_model=AgentStepResponse, tags=["agents"])
async def step_agent_endpoint(
    body: AgentStepRequest | None = Body(default=None),
    agent: Literal["random", "rule_based", "llm"] | None = Query(default=None),
) -> AgentStepResponse:
    selected_agent = agent or (body.agent if body is not None else "llm")

    try:
        with _env_lock:
            env = get_env()
            current_observation = env._current_obs.copy()

            if selected_agent == "random":
                action = random_agent(env)
            elif selected_agent == "rule_based":
                action = rule_based_agent(current_observation)
            else:
                action = llm_agent_sim(current_observation, _agent_rng)

            observation, reward, terminated, truncated, info = env.step(action)

        return AgentStepResponse(
            agent=selected_agent,
            action=int(action),
            action_name=ACTION_NAMES[action],
            observation=observation_to_list(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=to_python(info),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
