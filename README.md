# Commons Harvest Simulation with LLM Policymaker

This project implements a simulation of a **Tragedy of the Commons** scenario with two types of policymakers:

- A **rule-based policymaker**
- An **LLM-based policymaker**

The goal is to study how different policy strategies affect long-term **field health (social welfare)** in a shared-resource environment.

This implementation is based on the assignment specification: [assignment.pdf](./assignment.pdf)

## Overview

We simulate a shared apple field with:

- 10 agents
- Each agent decides how aggressively to harvest each round
- Overharvesting reduces field health
- If field health drops below 30%, all rewards are halved until recovery

A **policymaker sets a tax rate** each round to influence agent behavior.

### Objective

Maximize **field health over time** while maintaining sufficient rewards so that agents continue participating.

### Primary Metric

Average field health over the final 50 rounds

## Results (across 5 seeds)

![mean field health over time](./results/field_health.svg) ![mean agent reward over time](./results/avg_reward.svg)

#### Average field health over the final 50 rounds:

| Seed | Rule-Based | LLM-Based |
|------|------------|-----------|
|   8  |   0.752    |   0.406   |
|  39  |   0.713	|   0.856   |
|  45  |   0.754	|   0.790   |
|  69  |   0.000    |   0.855   |
| 224  |   0.719    |   0.831   |
| Mean |   0.588    |   0.748   |

### Insights

- The LLM policy-maker performs better overall
- The rule-based policy-maker succumbed to the tragedy. Barring that run, the rule-based policy-maker has more care for agent rewards (and performs better on that front). The LLMs don't have any real sense of what is a "decent" agent reward
- The LLMs generally stick around whichever tax rate they happened to pick first
- I suspect both the above issues can be addressed by using a LLM of more capacity

## Simulation Details

### Agents

- Each agent has a fixed **greed level ∈ [0,1]**
- Chooses between:
  - Gentle harvest (20% of fair share)
  - Aggressive harvest (100% of fair share)
- Uses a **softmax decision rule**
- Includes:
  - Revenue term
  - Convex cost of harvesting

Implementation details in: [simulator/core.py](./simulator/core.py)

### Environment Dynamics

- Field health evolves as: `H_next = H - total_harvest + regrowth`
- Regrowth follows a logistic form: `regrowth = r * H * (1 - H)`
- If field health < 0.3: All rewards are halved

### Policymakers

#### Rule-Based Policymaker

- Adjusts tax based on:
  - Deviation from target field health
  - Minimum acceptable reward
- Uses **small, smooth updates** to avoid instability

Implementation: [simulator/policy_maker.py](./simulator/policy_maker.py)

#### LLM-Based Policymaker

- Uses a structured prompt ([template](./simulator/prompt.txt)) with:
  1. Environment description
  2. Last 5 rounds of history
  3. Decision query
- Queries an LLM (`qwen2.5:1.5b`) via Ollama API
- Parses structured JSON response for tax rate

### Design Choices

- Introduced convex cost of harvesting to discourage over-exploitation
- Modeled greed as a softmax temperature, not an additive bias
- Used tax redistribution to maintain participation incentives
- Implemented robust parsing of LLM outputs to handle malformed responses

## Setup

This project uses **uv** for package management. Install with: `pip install uv`. Then run:

```bash
uv sync
source .venv/bin/activate
```
## Running Experiments

### Single Run

#### Rule-based policymaker:

```bash
python scripts/simulate.py --seed 42
```

#### LLM-based policymaker:

```bash
python scripts/simulate.py --use-llm --seed 42
```
### Batch Runs (5 Seeds)
```bash
bash scripts/run_experiments.sh
```

This runs both policymakers across 5 different seeds.

Script reference: [scripts/run_experiments.sh](./scripts/run_experiments.sh)

## Project Structure

```
.
├── simulator/
│   ├── core.py              # Simulation + Agent logic
│   ├── policy_maker.py      # Rule-based + LLM policymakers
│   └── prompt.txt           # LLM prompt template
│
├── scripts/
│   ├── simulate.py          # Main experiment runner
│   └── run_experiments.sh   # Batch experiment script
│
├── pyproject.toml
```

## Logging
Uses Weights & Biases (wandb) by default
Logs:
- Field health
- Tax rate
- Average harvest
- Average reward

Disable logging by passing: `--disable-wandb`

## Limitations and Future Work

- Agents do not learn over time (static behavior)
- Discrete hand designed harvest choices
- Use of a more capable LLM
- LLM has limited memory (last 5 rounds only)
- No explicit modeling of agent dropout