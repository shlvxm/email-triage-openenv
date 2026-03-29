import random
import uuid
from typing import Dict, List, Tuple
import numpy as np
from models import EmailAction, EmailObservation, EmailState, StepResult

# ── Environment Constants ───────────────────────────────────────────────────
ACTION_NAMES = {
    EmailAction.SPAM:    "spam",
    EmailAction.PRIMARY: "primary",
    EmailAction.SOCIAL:  "social",
    EmailAction.PROMO:   "promo",
    EmailAction.URGENT:  "urgent",
    EmailAction.REPLY:   "reply",
}

ACTION_EMOJIS = {
    EmailAction.SPAM:    "🗑️",
    EmailAction.PRIMARY: "📬",
    EmailAction.SOCIAL:  "👥",
    EmailAction.PROMO:   "🏷️",
    EmailAction.URGENT:  "🚨",
    EmailAction.REPLY:   "↩️",
}

OBS_DIM = 10

_TEMPLATES = {
    EmailAction.SPAM: [
        dict(urgency=(0.1,0.3),keyword=(0.0,0.2),sender=(0.0,0.2),
             spam=(0.75,1.0),promo=(0.1,0.4),social=(0.0,0.1),
             reply=(0.0,0.2),attach=(0.0,0.3),tod=(0.0,1.0),thread=(0.0,0.1)),
    ],
    EmailAction.PRIMARY: [
        dict(urgency=(0.3,0.6),keyword=(0.5,0.9),sender=(0.6,1.0),
             spam=(0.0,0.1),promo=(0.0,0.15),social=(0.0,0.1),
             reply=(0.2,0.6),attach=(0.3,0.9),tod=(0.3,0.75),thread=(0.0,0.5)),
    ],
    EmailAction.SOCIAL: [
        dict(urgency=(0.0,0.2),keyword=(0.1,0.4),sender=(0.2,0.6),
             spam=(0.0,0.1),promo=(0.1,0.3),social=(0.7,1.0),
             reply=(0.0,0.2),attach=(0.0,0.1),tod=(0.0,1.0),thread=(0.0,0.3)),
    ],
    EmailAction.PROMO: [
        dict(urgency=(0.0,0.3),keyword=(0.3,0.7),sender=(0.1,0.5),
             spam=(0.1,0.4),promo=(0.7,1.0),social=(0.0,0.2),
             reply=(0.0,0.1),attach=(0.0,0.2),tod=(0.1,0.9),thread=(0.0,0.2)),
    ],
    EmailAction.URGENT: [
        dict(urgency=(0.75,1.0),keyword=(0.6,1.0),sender=(0.5,1.0),
             spam=(0.0,0.1),promo=(0.0,0.2),social=(0.0,0.1),
             reply=(0.3,0.7),attach=(0.1,0.7),tod=(0.2,0.8),thread=(0.0,0.4)),
    ],
    EmailAction.REPLY: [
        dict(urgency=(0.4,0.75),keyword=(0.5,0.9),sender=(0.6,1.0),
             spam=(0.0,0.05),promo=(0.0,0.1),social=(0.0,0.15),
             reply=(0.7,1.0),attach=(0.0,0.5),tod=(0.3,0.8),thread=(0.1,0.8)),
    ],
}

_LEVEL_CONFIG = {
    "easy":   dict(noise_std=0.03, overlap_prob=0.05, ambiguous_prob=0.00),
    "medium": dict(noise_std=0.08, overlap_prob=0.15, ambiguous_prob=0.10),
    "hard":   dict(noise_std=0.15, overlap_prob=0.30, ambiguous_prob=0.25),
}

_REWARD_MATRIX = np.array([
    [1.00,0.00,0.05,0.05,0.00,0.00],
    [0.00,1.00,0.10,0.05,0.50,0.60],
    [0.05,0.10,1.00,0.15,0.00,0.05],
    [0.05,0.05,0.20,1.00,0.00,0.00],
    [0.00,0.40,0.00,0.00,1.00,0.70],
    [0.00,0.50,0.05,0.00,0.60,1.00],
], dtype=np.float32)

def generate_summary(obs, true_label):
    """Generate a pseudo-NL summary from raw features."""
    urg, kw, snd, spm, prm, soc, rep, att, tod, thr = obs
    names = ["low", "low-med", "medium", "med-high", "high"]
    val_to_name = lambda v: names[int(np.clip(v * 4.99, 0, 4))]
    
    summary = f"Email from sender (score: {val_to_name(snd)}) with {val_to_name(urg)} urgency. "
    if spm > 0.6: summary += "System flags high spam likelihood. "
    if prm > 0.6: summary += "Looks like a promotional campaign. "
    if soc > 0.6: summary += "Linked to social network activity. "
    if rep > 0.6: summary += "Sender is expecting a reply. "
    if att > 0.5: summary += "Contains attachments. "
    return summary

# ── Environment Class ────────────────────────────────────────────────────────
class EmailTriageEnv:
    def __init__(self, task_level: str = "medium", max_steps: int = 50, seed: int = None):
        if task_level not in _LEVEL_CONFIG:
            raise ValueError(f"Invalid task_level. Choose from {list(_LEVEL_CONFIG.keys())}")
        self.task_level = task_level
        self.max_steps = max_steps
        self.episode_id = str(uuid.uuid4())
        self._cfg = _LEVEL_CONFIG[task_level]
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        
        self.step_count = 0
        self.total_reward = 0.0
        self.current_obs = None
        self.true_label = None
        self.done = False

    def reset(self, seed: int = None) -> EmailObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._py_rng = random.Random(seed)
        
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        self.true_label = self._sample_label()
        self.current_obs = self._sample_features(self.true_label)
        
        return EmailObservation(
            features=self.current_obs.tolist(),
            summary=generate_summary(self.current_obs, self.true_label)
        )

    def step(self, action: EmailAction) -> StepResult:
        if self.done:
            raise RuntimeError("Environment is finished. Please call reset().")
            
        reward = float(_REWARD_MATRIX[self.true_label, action.value])
        self.total_reward += reward
        self.step_count += 1
        
        info = {
            "true_label": int(self.true_label),
            "predicted_action": int(action),
            "step_reward": reward,
            "is_correct": action.value == self.true_label
        }
        
        self.done = self.step_count >= self.max_steps
        
        if not self.done:
            self.true_label = self._sample_label()
            self.current_obs = self._sample_features(self.true_label)
        
        obs_model = EmailObservation(
            features=self.current_obs.tolist(),
            summary=generate_summary(self.current_obs, self.true_label)
        )
        
        return StepResult(
            observation=obs_model,
            reward=reward,
            done=self.done,
            info=info
        )

    def state(self) -> EmailState:
        return EmailState(
            step_count=self.step_count,
            episode_id=self.episode_id,
            max_steps=self.max_steps,
            true_label=int(self.true_label),
            total_reward=self.total_reward
        )

    def _sample_label(self) -> int:
        return self._py_rng.choices(range(6), weights=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10], k=1)[0]

    def _sample_features(self, label: int) -> np.ndarray:
        cfg = self._cfg
        tmpl = self._py_rng.choice(_TEMPLATES[label])
        blend_tmpl, alpha = None, 0.0
        if self._py_rng.random() < cfg["overlap_prob"]:
            bl = self._py_rng.choice([l for l in range(6) if l != label])
            blend_tmpl = self._py_rng.choice(_TEMPLATES[bl])
            alpha = self._py_rng.uniform(0.15, 0.35)
        
        keys = ["urgency","keyword","sender","spam","promo","social","reply","attach","tod","thread"]
        obs = np.empty(OBS_DIM, dtype=np.float32)
        for i, k in enumerate(keys):
            lo, hi = tmpl[k]
            val = self._py_rng.uniform(lo, hi)
            if blend_tmpl:
                blo, bhi = blend_tmpl[k]
                val = (1-alpha)*val + alpha*self._py_rng.uniform(blo, bhi)
            obs[i] = val
            
        obs = np.clip(obs + self._rng.normal(0, cfg["noise_std"], OBS_DIM), 0, 1).astype(np.float32)
        if self._py_rng.random() < cfg["ambiguous_prob"]:
            obs = (obs + 0.5) / 2.0
        return obs
