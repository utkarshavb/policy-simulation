import argparse
from simulator.core import CommonsSim
from simulator.policy_maker import RuleBasedPolicymaker

parser = argparse.ArgumentParser()
# state parameters
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
# growth rate and cost
parser.add_argument("--regrowth-rate", type=float, default=0.1, help="Regrowth rate of the common resource")
parser.add_argument("--cost-multiplier", type=float, default=5, help="Scaling factor for harvest cost")
# policy-maker parameters
parser.add_argument("--target-health", type=float, default=0.7, help="Target health level for the resource")
parser.add_argument("--target-harvest", type=float, default=0.45, help="Target average harvest level for the agents")
parser.add_argument("--reward-floor", type=float, default=0.08, help="If average reward falls below this level, the policymaker will increase taxes")
parser.add_argument("--k-h", type=float, default=0.6, help="Weight for health deviation in policymaker's tax adjustment")
parser.add_argument("--k-a", type=float, default=0.3, help="Weight for harvest deviation in policymaker's tax adjustment")
parser.add_argument("--k-r", type=float, default=0.4, help="Weight for reward deviation in policymaker's tax adjustment")
args = parser.parse_args()

n_agents = args.n_agents
seed = args.seed
regrowth_rate = args.regrowth_rate

policymaker = RuleBasedPolicymaker(
    args.target_health, args.target_harvest, args.reward_floor, args.k_h, args.k_a, args.k_r
)
sim = CommonsSim(
    n_agents, seed, regrowth_rate=regrowth_rate, cost_multiplier=args.cost_multiplier
)
obs = sim.reset()

greed_str = ", ".join([f"{agent.greed:.3f}" for agent in sim.agents])
print(f"greed levels of agents: {greed_str}")

for round in range(args.steps):
    tax_rate = policymaker.act(**obs)
    obs, field_health, info = sim.step(tax_rate)
    print(f"| {round=:3} | Field Health: {obs['field_health']:.3f} | Avg Harvest: {obs['avg_harvest']:.3f} | Avg Reward: {obs['avg_reward']:.5f} | Tax Rate: {tax_rate:.3f} |")