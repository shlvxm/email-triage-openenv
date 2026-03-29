import numpy as np
import random
from env import EmailTriageEnv
from models import EmailAction
from tasks import TASKS, Grader

# ── Baseline Agents ─────────────────────────────────────────────────────────

def random_agent(obs) -> EmailAction:
    return EmailAction(random.randint(0, 5))

def rule_based_agent(obs) -> EmailAction:
    # obs.features is a list of 10 floats
    f = obs.features
    u, kw, snd, spm, prm, soc, rep = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    
    if spm > 0.60 and snd < 0.25:  return EmailAction.SPAM
    if u  > 0.70 and snd > 0.45:  return EmailAction.URGENT
    if rep> 0.60 and snd > 0.55 and spm < 0.15: return EmailAction.REPLY
    if soc> 0.55 and prm < 0.40:   return EmailAction.SOCIAL
    if prm > 0.55:                  return EmailAction.PROMO
    if spm > 0.45:                  return EmailAction.SPAM
    return EmailAction.PRIMARY

def llm_agent_sim(obs) -> EmailAction:
    """Calibrated LLM simulation (as seen in monolithic version)."""
    f = obs.features
    u, kw, snd, spm, prm, soc, rep = f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    
    scores = {
        EmailAction.SPAM:    spm*0.80 + (1-snd)*0.20,
        EmailAction.URGENT:  u*0.65  + kw*0.20 + snd*0.15,
        EmailAction.REPLY:   rep*0.60 + snd*0.25 + kw*0.15,
        EmailAction.SOCIAL:  soc*0.80 + (1-u)*0.10 + (1-rep)*0.10,
        EmailAction.PROMO:   prm*0.75  + (1-snd)*0.15 + (1-u)*0.10,
        EmailAction.PRIMARY: kw*0.35  + snd*0.40 + (1-spm)*0.25,
    }
    # Add minor noise to simulate LLM variance
    for k in scores: scores[k] += random.gauss(0, 0.04)
    return max(scores, key=scores.get)

# ── Evaluation Script ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  OPENENV EMAIL TRIAGE BASELINE RESULTS  ")
    print("=" * 60)
    print(f"{'Task':<15} | {'Random':^10} | {'Rule-Based':^12} | {'LLM-Agent':^10}")
    print("-" * 60)

    for level, task in TASKS.items():
        grader = Grader(task)
        
        score_rand = grader.score(random_agent)
        score_rule = grader.score(rule_based_agent)
        score_llm  = grader.score(llm_agent_sim)
        
        print(f"{level:<15} | {score_rand:^10.2f} | {score_rule:^12.2f} | {score_llm:^10.2f}")

    print("-" * 60)
    print("Scores are in range [0.0, 1.0] (Higher is better)")
    print("=" * 60)

if __name__ == "__main__":
    main()
