import argparse
from simulator.core import CommonsSim
from simulator.policy_maker import RuleBasedPolicymaker

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--regrowth-rate", type=float, default=0.1, help="Regrowth rate of the common resource")
parser.add_argument("--min-resource", type=float, default=0.05, help="If resource falls below this level, regrowth is calculated using this value to prevent collapse")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

n_agents = args.n_agents
seed = args.seed
regrowth_rate = args.regrowth_rate
epsilon = args.min_resource

policymaker = RuleBasedPolicymaker()
sim = CommonsSim(n_agents, seed, regrowth_rate=regrowth_rate, epsilon=epsilon)
obs = sim.reset()

for round in range(args.steps):
    tax_rate = policymaker.act(**obs)
    obs, field_health, info = sim.step(tax_rate)
    print(f"| {round=:3} | Field Health: {obs['field_health']:.3f} | Avg Harvest: {obs['avg_harvest']:.3f} | Avg Reward: {obs['avg_reward']:.5f} | Tax Rate: {tax_rate:.3f} |")