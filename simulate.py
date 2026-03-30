import argparse
from simulator.core import CommonsSim
from simulator.policy_maker import RuleBasedPolicymaker, LLMPolicyMaker

parser = argparse.ArgumentParser()
# state parameters
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use-llm", action="store_true", help="Whether to use the LLM-based policymaker instead of the rule-based one")
# growth rate and cost
parser.add_argument("--regrowth-rate", type=float, default=1.45, help="Regrowth rate of the common resource")
parser.add_argument("--cost-multiplier", type=float, default=5, help="Scaling factor for harvest cost")
# rules-based policy-maker parameters
parser.add_argument("--target-health", type=float, default=0.5, help="Target health level for the resource")
parser.add_argument("--target-harvest", type=float, default=0.04, help="Target average harvest level for the agents")
parser.add_argument("--min-reward", type=float, default=0.015, help="If average reward falls below this level, the policymaker will increase taxes")
parser.add_argument("--health-wt", type=float, default=0.6, help="Weight for health deviation in policymaker's tax adjustment")
parser.add_argument("--harvest-wt", type=float, default=0.3, help="Weight for harvest deviation in policymaker's tax adjustment")
parser.add_argument("--reward-wt", type=float, default=0.4, help="Weight for reward deviation in policymaker's tax adjustment")
# llm-based policy-maker parameters
parser.add_argument("--model", type=str, default="llama3", help="LLM model to use for policymaking")
parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Base URL for Ollama API")
parser.add_argument("--prompt-file", type=str, default="simulator/prompt.txt", help="Path to prompt template file for LLM policymaker")
parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for LLM response generation")
args = parser.parse_args()

if not args.use_llm:
    policymaker = RuleBasedPolicymaker(
        args.target_health, args.target_harvest, args.min_reward,
        args.health_wt, args.harvest_wt, args.reward_wt
    )
else:
    policymaker = LLMPolicyMaker(
        model=args.model, base_url=args.base_url, prompt_file=args.prompt_file,
        temperature=args.temperature
    )

sim = CommonsSim(
    n_agents=args.n_agents, seed=args.seed, regrowth_rate=args.regrowth_rate,
    cost_multiplier=args.cost_multiplier
)
obs = sim.reset()
greed_str = ", ".join([f"{agent.greed:.3f}" for agent in sim.agents])
print(f"greed levels of agents: {greed_str}")

for round in range(args.steps):
    tax_rate = policymaker.act(**obs)
    obs, field_health, info = sim.step(tax_rate)
    print(f"| {round=:3} | Field Health: {obs['field_health']:.3f} | Avg Harvest: {obs['avg_harvest']:.3f} | Avg Reward: {obs['avg_reward']:.5f} | Tax Rate: {tax_rate:.3f} |")