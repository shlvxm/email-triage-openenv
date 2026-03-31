---
title: Email Triage RL Agent
emoji: 📬
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
license: mit
short_description: RL env for Indian email triage with openenv protocol
---

# Email Triage RL Agent

`antigravity` is a Docker-based Hugging Face Space that serves a self-contained Gymnasium environment called `EmailTriageEnv`. It exposes OpenEnv-style HTTP endpoints for resetting the environment, stepping through it, and benchmarking built-in baseline agents.

## Environment Summary

- Observation space: `Box(10,)`
- Action space: `Discrete(6)`
- Actions:
  - `0 = spam`
  - `1 = primary`
  - `2 = social`
  - `3 = promo`
  - `4 = urgent`
  - `5 = reply`
- Task levels: `easy`, `medium`, `hard`
- Reward range: `0.0` to `1.0`
- Indian context: IRCTC alerts, Flipkart promos, EMI bounce warnings, ITR reminders, manager follow-ups

## Observation Features

The observation vector is ordered as:

| Index | Feature | Meaning |
| --- | --- | --- |
| 0 | `urgency` | How time-sensitive the email is |
| 1 | `keyword` | Strength of task-relevant keywords |
| 2 | `sender` | Trust and importance of sender |
| 3 | `spam` | Spam likelihood |
| 4 | `promo` | Promotional intent |
| 5 | `social` | Social-notification signal |
| 6 | `reply` | Likelihood that a reply is expected |
| 7 | `attachment` | Attachment/document signal |
| 8 | `time_of_day` | Encoded timing clue |
| 9 | `thread_length` | Ongoing conversation depth |

## Reward Matrix

Rewards are shaped, not binary. Correct actions receive `1.0`, while near-miss actions receive partial credit depending on the class overlap.

Rows are ground-truth labels and columns are chosen actions in this order:
`[spam, primary, social, promo, urgent, reply]`

| True \ Pred | spam | primary | social | promo | urgent | reply |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spam | 1.00 | 0.10 | 0.05 | 0.25 | 0.00 | 0.00 |
| primary | 0.00 | 1.00 | 0.15 | 0.10 | 0.45 | 0.35 |
| social | 0.05 | 0.15 | 1.00 | 0.15 | 0.05 | 0.10 |
| promo | 0.10 | 0.10 | 0.20 | 1.00 | 0.05 | 0.00 |
| urgent | 0.00 | 0.35 | 0.05 | 0.00 | 1.00 | 0.55 |
| reply | 0.00 | 0.25 | 0.05 | 0.00 | 0.60 | 1.00 |

## Baseline Results

Approximate average reward by task level:

| Agent | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| Random | 0.31 | 0.25 | 0.20 |
| Rule-Based | 0.72 | 0.65 | 0.58 |
| LLM-Agent | 0.88 | 0.82 | 0.76 |

## API

### `GET /`

Health check.

Response:

```json
{"status": "ok"}
```

### `GET /health`

Health check used by Docker and Hugging Face Spaces.

Response:

```json
{"status": "healthy"}
```

### `GET /info`

Returns environment metadata, feature names, task levels, reward matrix, and baseline scores.

### `GET /metadata`

Returns OpenEnv metadata including environment name and description.

### `GET /schema`

Returns action, observation, and state schemas for validator compatibility.

### `GET /state`

Returns the current live environment state, including the active observation and episode progress.

### `POST /reset`

Resets the environment and returns a fresh observation.

Accepted body:

```json
{}
```

Optional body:

```json
{
  "seed": 42,
  "task_level": "medium",
  "options": {}
}
```

Response schema:

```json
{
  "observation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "info": {}
}
```

### `POST /step`

Takes one action in the current episode.

Request:

```json
{"action": 4}
```

Response schema:

```json
{
  "observation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "reward": 1.0,
  "terminated": false,
  "truncated": false,
  "info": {
    "task_level": "medium",
    "step": 1,
    "true_label": 4,
    "true_label_name": "urgent",
    "action_name": "urgent",
    "scenario_id": "urgent-irctc",
    "scenario_subject": "IRCTC waitlist updated for tonight's journey",
    "scenario_context": "Travel alert that needs immediate attention.",
    "episode_reward": 1.0
  }
}
```

Invalid actions such as `{"action": 6}` return HTTP `422`.

### `POST /step/agent`

Auto-acts using a built-in baseline policy.

Accepted body:

```json
{"agent": "llm"}
```

Allowed agents:

- `random`
- `rule_based`
- `llm`

The endpoint also accepts an optional query parameter, for example:

```bash
curl -X POST "http://localhost:7860/step/agent?agent=rule_based"
```

## Indian Email Context Examples

- `IRCTC waitlist updated for tonight's journey`
- `EMI payment failed, avoid penalty charges`
- `ITR filing deadline reminder for FY 2025-26`
- `Flipkart Big Saving Days starts tonight`
- `Need your update before the 4 PM review`

## Run Locally

Build the Docker image:

```bash
docker build -t antigravity .
```

Run the Space locally:

```bash
docker run --rm -p 7860:7860 antigravity
```

Then test the OpenEnv endpoints:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/metadata
curl http://localhost:7860/schema
curl http://localhost:7860/state
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"action\": 0}"
curl -X POST http://localhost:7860/step/agent -H "Content-Type: application/json" -d "{\"agent\": \"llm\"}"
curl -X POST http://localhost:7860/mcp -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}"
```

## Deployment Notes

- SDK: `docker`
- Container port: `7860`
- Entrypoint: `python inference.py`
- Server: FastAPI + Uvicorn
- The environment is fully self-contained inside `inference.py`
