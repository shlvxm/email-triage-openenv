#!/usr/bin/env python3
"""
Dual-mode inference.py for OpenEnv Hackathon
- CLI mode: Outputs structured [START]/[STEP]/[END] blocks
- Server mode: Runs FastAPI web server for Hugging Face Space
"""
from __future__ import annotations

import json
import os
import sys
import random
import socket
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from openai import OpenAI
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

# Hackathon LLM proxy config (injected by validator)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY")

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


def observation_to_list(obs: np.ndarray | list[float]) -> list[float]:
    if isinstance(obs, np.ndarray):
        return [float(x) for x in obs.tolist()]
    return [float(x) for x in obs]


class EmailTriageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_level: str = DEFAULT_TASK_LEVEL, max_steps: int = MAX_STEPS, seed: int | None = None):
        super().__init__()
        if task_level not in TASK_LEVELS:
            raise ValueError(f"task_level must be one of {TASK_LEVELS}, got {task_level!r}")
        self.task_level = task_level
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_obs = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self._current_label = 0
        self._current_scenario = SCENARIOS[0]
        self._needs_reset = True

    def _generate_observation(self) -> tuple[np.ndarray, int, dict]:
        label = self._rng.choice(len(ACTION_NAMES), p=LABEL_WEIGHTS)
        scenario = self._rng.choice(SCENARIOS_BY_LABEL[label])
        base = np.array(scenario["features"], dtype=np.float32)
        config = LEVEL_CONFIG[self.task_level]
        obs = base + self._rng.normal(0, config["noise_std"], size=len(FEATURE_NAMES))
        if self._rng.random() < config["blend_prob"]:
            confuser_label = self._rng.choice(CONFUSER_LABELS[label])
            confuser = self._rng.choice(SCENARIOS_BY_LABEL[confuser_label])
            alpha = self._rng.uniform(*config["blend_alpha"])
            obs = obs * (1 - alpha) + np.array(confuser["features"], dtype=np.float32) * alpha
        if self._rng.random() < config["flip_prob"]:
            label = self._rng.choice(CONFUSER_LABELS[label])
        obs = np.clip(obs, 0.0, 1.0)
        return obs, label, scenario

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "task_level" in options:
            new_level = options["task_level"]
            if new_level not in TASK_LEVELS:
                raise ValueError(f"task_level must be one of {TASK_LEVELS}, got {new_level!r}")
            self.task_level = new_level
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_obs, self._current_label, self._current_scenario = self._generate_observation()
        self._needs_reset = False
        info = {
            "task_level": self.task_level,
            "scenario_id": self._current_scenario["id"],
            "true_label": int(self._current_label),
            "true_label_name": ACTION_NAMES[self._current_label],
        }
        return self._current_obs.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._needs_reset:
            raise RuntimeError("Environment needs reset. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in range [0, {len(ACTION_NAMES)-1}]")
        reward = float(REWARD_MATRIX[self._current_label, action])
        self._episode_reward += reward
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        if not truncated:
            self._current_obs, self._current_label, self._current_scenario = self._generate_observation()
        else:
            self._needs_reset = True
        info = {
            "true_label": int(self._current_label),
            "true_label_name": ACTION_NAMES[self._current_label],
            "action_name": ACTION_NAMES[action],
            "scenario_id": self._current_scenario["id"],
            "step": self._step_count,
            "episode_reward": float(self._episode_reward),
        }
        return self._current_obs.copy(), reward, terminated, truncated, info


_env: EmailTriageEnv | None = None
_env_lock = threading.Lock()


def get_env() -> EmailTriageEnv:
    global _env
    if _env is None:
        _env = EmailTriageEnv(task_level=DEFAULT_TASK_LEVEL, max_steps=MAX_STEPS)
        _env.reset()
    return _env


_agent_rng = np.random.default_rng(42)


def random_agent(env: EmailTriageEnv) -> int:
    return env.action_space.sample()


def rule_based_agent(obs: np.ndarray) -> int:
    urgency, keyword, sender, spam, promo, social, reply, attachment, time_of_day, thread_length = obs
    if spam > 0.7:
        return 0
    if reply > 0.8:
        return 5
    if urgency > 0.8 and sender > 0.7:
        return 4
    if social > 0.8:
        return 2
    if promo > 0.8:
        return 3
    return 1


def llm_agent_sim(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Call the hackathon LLM proxy using OpenAI client to decide the email action."""
    feature_values = dict(zip(FEATURE_NAMES, [float(x) for x in obs]))
    prompt_lines = [
        "You are an email triage agent. Given email feature scores (0.0-1.0), pick the best action.",
        "",
        "Features: " + json.dumps(feature_values),
        "",
        "Actions: 0=spam, 1=primary, 2=social, 3=promo, 4=urgent, 5=reply",
        "",
        "Reply with ONLY a single digit 0-5. No explanation.",
    ]
    prompt = "\n".join(prompt_lines)

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY if API_KEY else "dummy-key",
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        action = int(text[0])
        if 0 <= action <= 5:
            return action
    except Exception:
        pass

    # Fallback: rule-based heuristic
    urgency, keyword, sender, spam, promo, social, reply, attachment, time_of_day, thread_length = obs
    scores = np.array(
        [
            spam * 0.9 + (1 - sender) * 0.6 + urgency * 0.2,
            sender * 0.7 + (1 - spam) * 0.5 + attachment * 0.3,
            social * 0.95 + (1 - urgency) * 0.2,
            promo * 0.9 + keyword * 0.3,
            urgency * 0.9 + sender * 0.5 + (1 - spam) * 0.4,
            reply * 0.95 + thread_length * 0.4 + urgency * 0.3,
        ],
        dtype=np.float32,
    )
    scores += rng.normal(0, 0.05, size=len(ACTION_NAMES))
    return int(np.argmax(scores))


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class ResetRequest(BaseModel):
    task_level: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    options: dict[str, Any] | None = Field(default=None)


class ResetResponse(BaseModel):
    observation: list[float]
    info: dict[str, Any]


class StepRequest(BaseModel):
    action: int = Field(ge=0, lt=len(ACTION_NAMES))


class StepResponse(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: dict[str, Any]


class AgentStepRequest(BaseModel):
    agent: Literal["random", "rule_based", "llm"] = Field(default="llm")


class AgentStepResponse(BaseModel):
    agent: str
    action: int
    action_name: str
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: dict[str, Any]


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    get_env()
    yield
    global _env
    with _env_lock:
        if _env is not None:
            _env.close()
            _env = None


app = FastAPI(
    title="Email Triage RL Environment",
    description="OpenEnv-compatible Gymnasium environment for email triage with Indian context",
    version="1.0.0",
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
            done=bool(terminated or truncated),
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
            done=bool(terminated or truncated),
            info=to_python(info),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


def run_cli_inference():
    """
    CLI mode for OpenEnv validation
    Outputs structured [START]/[STEP]/[END] blocks
    """
    # Default configuration
    task_name = "email_triage"
    task_level = DEFAULT_TASK_LEVEL
    agent_type = "llm"  # Can be: random, rule_based, llm
    max_steps = MAX_STEPS
    seed = 42
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
            task_level = config.get("task_level", task_level)
            agent_type = config.get("agent_type", agent_type)
            max_steps = config.get("max_steps", max_steps)
            seed = config.get("seed", seed)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Print START block
    print(f"[START] task={task_name}", flush=True)
    
    # Create environment
    env = EmailTriageEnv(task_level=task_level, max_steps=max_steps, seed=seed)
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    steps = 0
    rng = np.random.default_rng(seed)
    
    # Run episode
    for step in range(max_steps):
        # Choose action based on agent type
        if agent_type == "random":
            action = random_agent(env)
        elif agent_type == "rule_based":
            action = rule_based_agent(obs)
        else:  # llm
            action = llm_agent_sim(obs, rng)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print step info
        print(f"[STEP] step={steps} reward={reward:.2f}", flush=True)
        
        if terminated or truncated:
            break
    
    # Calculate final score
    score = total_reward / steps if steps > 0 else 0.0
    
    # Print END block
    print(f"[END] task={task_name} score={score:.2f} steps={steps}", flush=True)
    
    env.close()


if __name__ == "__main__":
    # Always run CLI inference - validator runs inference.py directly for [START]/[STEP]/[END] output.
    # The validator manages the HTTP server separately; do NOT start uvicorn here.
    run_cli_inference()
