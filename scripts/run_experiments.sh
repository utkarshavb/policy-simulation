#!/bin/bash

# number of seeds
SEEDS=(8 39 45 69 224)

echo "Running RULE-BASED policymaker experiments..."
for SEED in "${SEEDS[@]}"
do
  python scripts/simulate.py \
    --seed $SEED \
    --run rule_seed${SEED}
done

echo "Running LLM policymaker experiments..."
for SEED in "${SEEDS[@]}"
do
  python scripts/simulate.py \
    --use-llm \
    --seed $SEED \
    --run llm_seed${SEED}
done

echo "All experiments completed."