---
title: Email Triage RL Agent (OpenEnv)
emoji: 📬
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
license: mit
short_description: OpenEnv-compliant RL environment for Indian inbox triage.
tags: [openenv, reinforcement-learning, gymnasium]
---

# 📬 Email Triage OpenEnv

> **Standardised RL Environment for Inbox Management**  
> Indian professional context: IRCTC, Flipkart/Swiggy promos, EMI notices, Manager follow-ups.

This project implements a complete, real-world **OpenEnv** compliant environment. It provides a structured interface with typed Pydantic models, detailed reward shaping, and a suite of benchmarking tasks.

---

## 🏗️ OpenEnv Architecture

The environment adheres to the OpenEnv specification for agentic reinforcement learning:

```
EmailTriage (OpenEnv)
│
├── reset() -> EmailObservation      # Initialises episode
├── step(action: EmailAction)        # Executes triage, returns StepResult
└── state() -> EmailState           # Accesses current episode metadata
```

### Typed Models (Pydantic)
- **EmailObservation**: Numerical feature vector (10-dim) + Natural language summary.
- **EmailAction**: Discrete triage categories (0-5).
- **EmailState**: Episode ID, step count, max steps, and total reward.

---

## 🔧 Tasks & Evaluation

We provide 3 standardized tasks with increasing difficulty for benchmarking agents:

| Task ID | Level | Characteristics |
|:---:|:---:|:---|
| `easy` | 🟢 Easy | Low noise, high class separation, 0% overlap. |
| `medium` | 🟡 Medium | 15% class overlap, moderate noise, 10% ambiguity. |
| `hard` | 🔴 Hard | 30% class overlap, high noise, 25% ambiguity. |

### Programmatic Grader
Each task is scored using a **Grader** that produces a scalar value `[0.0, 1.0]` based on:
- **Triage Accuracy**: Percentage of correct category assignments.
- **Reward Efficiency**: Average reward per step using the shaped matrix.

---

## 📊 Baseline Results

| Task Level | Random | Rule-Based | LLM-Agent |
|:---:|:---:|:---:|:---:|
| 🟢 easy   | 0.25 | 0.65 | 0.82 |
| 🟡 medium | 0.22 | 0.58 | 0.78 |
| 🔴 hard   | 0.20 | 0.52 | 0.75 |

Reproducible using: `python inference.py`

---

## 🚀 Quick Start

### Local Execution (Python)
```bash
pip install -r requirements.txt
streamlit run demo.py
```

### Running the Baseline Benchmark
```bash
python inference.py
```

### Docker Execution
```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv
```

---

## 📁 Project Structure

- `models.py`: Pydantic definitions for observations, actions, and states.
- `env.py`: The core `EmailTriageEnv` logic (OpenEnv compliant).
- `tasks.py`: Task definitions and benchmarking graders.
- `inference.py`: Baseline agent implementation and evaluation script.
- `demo.py`: Streamlit-based interactive dashboard.
- `openenv.yaml`: environment manifest for validation.
- `Dockerfile`: Container configuration for HF Spaces.

---

## ⚖️ License
MIT License. Built for the **Meta-Pytorch / Scaler OpenEnv Hackathon**.
