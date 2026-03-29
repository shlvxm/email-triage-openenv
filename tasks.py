from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
from env import EmailTriageEnv
from models import EmailAction

class Task(ABC):
    def __init__(self, name: str, level: str, episodes: int = 10):
        self.name = name
        self.level = level
        self.episodes = episodes

    @abstractmethod
    def get_env(self, seed: int = None) -> EmailTriageEnv:
        pass

class EmailTriageTask(Task):
    def __init__(self, name: str, level: str, episodes: int = 10, max_steps: int = 50):
        super().__init__(name, level, episodes)
        self.max_steps = max_steps

    def get_env(self, seed: int = None) -> EmailTriageEnv:
        return EmailTriageEnv(task_level=self.level, max_steps=self.max_steps, seed=seed)

class Grader:
    def __init__(self, task: Task):
        self.task = task

    def score(self, agent_fn) -> float:
        """
        Evaluate an agent on the task and return a score [0, 1].
        The score is a weighted average of accuracy and normalized total reward.
        """
        env = self.task.get_env()
        episode_scores = []

        for ep in range(self.task.episodes):
            obs = env.reset(seed=ep)
            done = False
            total_reward = 0.0
            correct_count = 0
            steps = 0
            
            while not done:
                action = agent_fn(obs)
                res = env.step(action)
                obs, reward, done, info = res.observation, res.reward, res.done, res.info
                
                total_reward += reward
                if info.get("is_correct"):
                    correct_count += 1
                steps += 1
                
            accuracy = correct_count / steps if steps > 0 else 0
            norm_reward = total_reward / steps if steps > 0 else 0
            
            # Final score for the episode is 60% accuracy + 40% reward signal
            ep_score = (0.6 * accuracy) + (0.4 * norm_reward)
            episode_scores.append(ep_score)
            
        return float(np.mean(episode_scores))

# Predefined Tasks
TASKS = {
    "easy": EmailTriageTask(name="Safe Triage (Easy)", level="easy"),
    "medium": EmailTriageTask(name="Enterprise Inbox (Medium)", level="medium"),
    "hard": EmailTriageTask(name="Agent Stress Test (Hard)", level="hard"),
}
