#!/usr/bin/env python3
"""
inference.py for OpenEnv Hackathon - Email Triage RL Agent
Outputs structured [START]/[STEP]/[END] blocks to stdout.
Uses the hackathon LLM proxy via OpenAI client.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from gymnasium import spaces
from openai import OpenAI
from pydantic import BaseModel, Field


# ── Feature / action definitions ────────────────────────────────────────────
FEATURE_NAMES = [
    "urgency", "keyword", "sender", "spam", "promo",
    "social", "reply", "attachment", "time_of_day", "thread_length",
]
ACTION_NAMES = ["spam", "primary", "social", "promo", "urgent", "reply"]
TASK_LEVELS  = ("easy", "medium", "hard")
LABEL_WEIGHTS = [0.16, 0.18, 0.14, 0.18, 0.17, 0.17]

REWARD_MATRIX = np.array([
    [1.00, 0.10, 0.05, 0.25, 0.00, 0.00],
    [0.00, 1.00, 0.15, 0.10, 0.45, 0.35],
    [0.05, 0.15, 1.00, 0.15, 0.05, 0.10],
    [0.10, 0.10, 0.20, 1.00, 0.05, 0.00],
    [0.00, 0.35, 0.05, 0.00, 1.00, 0.55],
    [0.00, 0.25, 0.05, 0.00, 0.60, 1.00],
], dtype=np.float32)

LEVEL_CONFIG = {
    "easy":   {"noise_std": 0.03, "blend_prob": 0.05, "blend_alpha": (0.08, 0.16), "flip_prob": 0.02},
    "medium": {"noise_std": 0.08, "blend_prob": 0.18, "blend_alpha": (0.12, 0.26), "flip_prob": 0.07},
    "hard":   {"noise_std": 0.14, "blend_prob": 0.32, "blend_alpha": (0.18, 0.38), "flip_prob": 0.14},
}

CONFUSER_LABELS = {0:[3,1], 1:[4,5], 2:[3,1], 3:[0,2], 4:[5,1], 5:[4,1]}

SCENARIOS = [
    {"id":"spam-bank-kyc","subject":"Urgent: Update your bank KYC immediately","label":0,"context":"Fake KYC warning.","features":[0.55,0.70,0.10,0.96,0.15,0.02,0.12,0.35,0.20,0.05]},
    {"id":"spam-crypto-giveaway","subject":"Claim your crypto giveaway before midnight","label":0,"context":"Classic scam.","features":[0.30,0.45,0.05,0.98,0.25,0.05,0.05,0.10,0.75,0.02]},
    {"id":"spam-lottery","subject":"You won a holiday package lottery","label":0,"context":"Promo scam.","features":[0.62,0.65,0.08,0.94,0.10,0.04,0.10,0.22,0.15,0.03]},
    {"id":"primary-electricity-bill","subject":"BESCOM monthly bill available","label":1,"context":"Billing email.","features":[0.48,0.55,0.82,0.04,0.08,0.05,0.18,0.42,0.35,0.15]},
    {"id":"primary-order-shipped","subject":"Your order has shipped","label":1,"context":"Order update.","features":[0.40,0.50,0.74,0.03,0.12,0.03,0.10,0.25,0.55,0.12]},
    {"id":"primary-bank-statement","subject":"Your account statement for March is ready","label":1,"context":"Bank statement.","features":[0.36,0.45,0.88,0.02,0.06,0.02,0.14,0.38,0.50,0.10]},
    {"id":"social-linkedin","subject":"You have a new LinkedIn invitation","label":2,"context":"LinkedIn notification.","features":[0.12,0.20,0.58,0.03,0.10,0.92,0.18,0.00,0.60,0.05]},
    {"id":"social-instagram","subject":"Someone mentioned you on Instagram","label":2,"context":"Instagram notification.","features":[0.10,0.18,0.46,0.05,0.08,0.95,0.09,0.00,0.72,0.04]},
    {"id":"social-alumni-group","subject":"Your alumni group has 12 new messages","label":2,"context":"Community chatter.","features":[0.18,0.24,0.64,0.02,0.06,0.88,0.22,0.05,0.66,0.18]},
    {"id":"promo-flipkart-sale","subject":"Flipkart Big Saving Days starts tonight","label":3,"context":"Ecommerce promo.","features":[0.22,0.58,0.52,0.08,0.97,0.05,0.04,0.02,0.62,0.06]},
    {"id":"promo-zomato-offer","subject":"Flat 60% off on dinner tonight","label":3,"context":"Food promo.","features":[0.18,0.44,0.48,0.06,0.93,0.06,0.03,0.01,0.78,0.04]},
    {"id":"promo-insurance-upsell","subject":"Upgrade your policy with bonus benefits","label":3,"context":"Marketing email.","features":[0.28,0.53,0.60,0.05,0.89,0.04,0.08,0.10,0.40,0.09]},
    {"id":"urgent-irctc","subject":"IRCTC waitlist updated for tonight's journey","label":4,"context":"Travel alert.","features":[0.95,0.92,0.90,0.02,0.06,0.01,0.15,0.08,0.32,0.05]},
    {"id":"urgent-emi-bounce","subject":"EMI payment failed, avoid penalty charges","label":4,"context":"Financial alert.","features":[0.98,0.88,0.86,0.03,0.05,0.00,0.22,0.18,0.58,0.22]},
    {"id":"urgent-itr-deadline","subject":"ITR filing deadline reminder for FY 2025-26","label":4,"context":"Tax deadline.","features":[0.93,0.94,0.84,0.02,0.04,0.01,0.18,0.12,0.40,0.10]},
    {"id":"reply-manager-followup","subject":"Need your update before the 4 PM review","label":5,"context":"Manager follow-up.","features":[0.82,0.86,0.94,0.01,0.03,0.04,0.97,0.22,0.68,0.82]},
    {"id":"reply-client-approval","subject":"Please confirm the revised proposal today","label":5,"context":"Client email.","features":[0.76,0.82,0.88,0.02,0.05,0.02,0.92,0.35,0.52,0.70]},
    {"id":"reply-recruiter","subject":"Can we reschedule your interview slot?","label":5,"context":"Recruiter outreach.","features":[0.70,0.78,0.90,0.03,0.04,0.05,0.89,0.18,0.45,0.60]},
]

SCENARIOS_BY_LABEL = {
    label: [s for s in SCENARIOS if s["label"] == label]
    for label in range(len(ACTION_NAMES))
}

# ── Environment config ───────────────────────────────────────────────────────
def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

DEFAULT_TASK_LEVEL = os.getenv("TASK_LEVEL", "medium").strip().lower()
if DEFAULT_TASK_LEVEL not in TASK_LEVELS:
    DEFAULT_TASK_LEVEL = "medium"

MAX_STEPS  = max(1, _get_env_int("MAX_STEPS", 10))
PORT       = _get_env_int("PORT", 7860)
SPACE_NAME = os.getenv("SPACE_ID", "email-triage-rl")

# ── Hackathon LLM proxy config (injected by validator) ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.getenv("API_KEY")


# ── Helpers ──────────────────────────────────────────────────────────────────
def to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_python(i) for i in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(i) for i in value]
    return value

def observation_to_list(obs: np.ndarray | list[float]) -> list[float]:
    if isinstance(obs, np.ndarray):
        return [float(x) for x in obs.tolist()]
    return [float(x) for x in obs]


# ── Gymnasium environment ────────────────────────────────────────────────────
class EmailTriageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_level: str = DEFAULT_TASK_LEVEL, max_steps: int = MAX_STEPS, seed: int | None = None):
        super().__init__()
        if task_level not in TASK_LEVELS:
            raise ValueError(f"task_level must be one of {TASK_LEVELS}")
        self.task_level   = task_level
        self.max_steps    = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space      = spaces.Discrete(len(ACTION_NAMES))
        self._rng              = np.random.default_rng(seed)
        self._step_count       = 0
        self._episode_reward   = 0.0
        self._current_obs      = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self._current_label    = 0
        self._current_scenario = SCENARIOS[0]
        self._needs_reset      = True

    def _generate_observation(self):
        label    = self._rng.choice(len(ACTION_NAMES), p=LABEL_WEIGHTS)
        scenario = self._rng.choice(SCENARIOS_BY_LABEL[label])
        base     = np.array(scenario["features"], dtype=np.float32)
        cfg      = LEVEL_CONFIG[self.task_level]
        obs      = base + self._rng.normal(0, cfg["noise_std"], size=len(FEATURE_NAMES))
        if self._rng.random() < cfg["blend_prob"]:
            cl  = self._rng.choice(CONFUSER_LABELS[label])
            cf  = self._rng.choice(SCENARIOS_BY_LABEL[cl])
            a   = self._rng.uniform(*cfg["blend_alpha"])
            obs = obs * (1 - a) + np.array(cf["features"], dtype=np.float32) * a
        if self._rng.random() < cfg["flip_prob"]:
            label = self._rng.choice(CONFUSER_LABELS[label])
        return np.clip(obs, 0.0, 1.0), label, scenario

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "task_level" in options:
            lv = options["task_level"]
            if lv not in TASK_LEVELS:
                raise ValueError(f"Invalid task_level: {lv}")
            self.task_level = lv
        self._step_count     = 0
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

    def step(self, action: int):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        reward             = float(REWARD_MATRIX[self._current_label, action])
        self._episode_reward += reward
        self._step_count   += 1
        terminated         = False
        truncated          = self._step_count >= self.max_steps
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


# ── Agents ───────────────────────────────────────────────────────────────────
def random_agent(env: EmailTriageEnv) -> int:
    return env.action_space.sample()

def rule_based_agent(obs: np.ndarray) -> int:
    urgency, keyword, sender, spam, promo, social, reply, attachment, time_of_day, thread_length = obs
    if spam > 0.7:   return 0
    if reply > 0.8:  return 5
    if urgency > 0.8 and sender > 0.7: return 4
    if social > 0.8: return 2
    if promo > 0.8:  return 3
    return 1

def llm_agent(obs: np.ndarray) -> int:
    """
    Call the hackathon LLM proxy via OpenAI client.
    Uses API_BASE_URL and API_KEY injected by the validator.
    """
    feature_values = dict(zip(FEATURE_NAMES, [float(x) for x in obs]))
    prompt = "\n".join([
        "You are an email triage agent. Given email feature scores (0.0-1.0), pick the best action.",
        "",
        "Features: " + json.dumps(feature_values),
        "",
        "Actions: 0=spam, 1=primary, 2=social, 3=promo, 4=urgent, 5=reply",
        "",
        "Reply with ONLY a single digit 0-5. No explanation.",
    ])

    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", API_BASE_URL),
        api_key=os.environ.get("API_KEY", API_KEY or "dummy"),
    )
    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", MODEL_NAME),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    for ch in text:
        if ch.isdigit() and 0 <= int(ch) <= 5:
            return int(ch)
    return 1


# ── FastAPI server ───────────────────────────────────────────────────────────
_env: EmailTriageEnv | None = None
_env_lock = threading.Lock()
_agent_rng = np.random.default_rng(42)

def get_env() -> EmailTriageEnv:
    global _env
    if _env is None:
        _env = EmailTriageEnv(task_level=DEFAULT_TASK_LEVEL, max_steps=MAX_STEPS)
        _env.reset()
    return _env

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

app = FastAPI(title="Email Triage RL Environment", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_model=HealthResponse)
async def root(): return HealthResponse(status="ok")

@app.get("/health", response_model=HealthResponse)
async def health(): return HealthResponse(status="healthy")

@app.get("/metadata")
async def metadata():
    return {"name": "Email Triage RL Agent", "version": "1.0.0", "mode": "http"}

@app.get("/schema")
async def schema():
    return {
        "action": {"type": "integer", "minimum": 0, "maximum": 5},
        "observation": {"type": "array", "items": {"type": "number"}, "feature_names": FEATURE_NAMES},
    }

@app.get("/state")
async def state():
    env = get_env()
    return {
        "task_level": env.task_level,
        "step": env._step_count,
        "max_steps": env.max_steps,
        "current_observation": observation_to_list(env._current_obs),
        "current_label": int(env._current_label),
        "current_label_name": ACTION_NAMES[env._current_label],
        "episode_reward": float(env._episode_reward),
        "needs_reset": bool(env._needs_reset),
    }

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(body: ResetRequest | None = Body(default=None)):
    payload = body or ResetRequest()
    try:
        with _env_lock:
            global _env
            requested_level = payload.task_level or (get_env().task_level if _env is not None else DEFAULT_TASK_LEVEL)
            if _env is None or _env.task_level != requested_level:
                if _env is not None:
                    _env.close()
                _env = EmailTriageEnv(task_level=requested_level, max_steps=MAX_STEPS, seed=payload.seed)
            options = dict(payload.options or {})
            if payload.task_level:
                options["task_level"] = payload.task_level
            observation, info = _env.reset(seed=payload.seed, options=options)
        return ResetResponse(observation=observation_to_list(observation), info=to_python(info))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/step", response_model=StepResponse)
async def step_endpoint(body: StepRequest):
    try:
        with _env_lock:
            env = get_env()
            observation, reward, terminated, truncated, info = env.step(body.action)
        return StepResponse(
            observation=observation_to_list(observation), reward=float(reward),
            terminated=bool(terminated), truncated=bool(truncated),
            done=bool(terminated or truncated), info=to_python(info),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/mcp")
async def mcp_endpoint(payload: dict[str, Any] | None = Body(default=None)):
    return {"jsonrpc": "2.0", "id": None if not payload else payload.get("id"),
            "error": {"code": -32601, "message": "MCP not implemented."}}


# ── CLI inference (for validator Phase 2) ────────────────────────────────────
def run_cli_inference():
    task_name  = "email_triage"
    task_level = DEFAULT_TASK_LEVEL
    max_steps  = MAX_STEPS
    seed       = 42

    if len(sys.argv) > 1:
        try:
            config     = json.loads(sys.argv[1])
            task_level = config.get("task_level", task_level)
            max_steps  = config.get("max_steps", max_steps)
            seed       = config.get("seed", seed)
        except (json.JSONDecodeError, ValueError):
            pass

    print(f"[START] task={task_name}", flush=True)

    env = EmailTriageEnv(task_level=task_level, max_steps=max_steps, seed=seed)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = llm_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"[STEP] step={steps} reward={reward:.4f}", flush=True)
        if terminated or truncated:
            break

    score = total_reward / steps if steps > 0 else 0.0
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)
    env.close()


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    run_cli_inference()
