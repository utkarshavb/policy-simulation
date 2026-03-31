import argparse
from simulator.core import CommonsSim
from simulator.policy_maker import RuleBasedPolicymaker, LLMPolicyMaker
import wandb

parser = argparse.ArgumentParser()
# state parameters
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run", type=str, default=None, help="WandB run name for logging")
parser.add_argument("--use-llm", action="store_true", help="Whether to use the LLM-based policymaker instead of the rule-based one")
parser.add_argument("--disable-wandb", action="store_true", help="Whether to disable WandB logging")
# growth rate and cost
parser.add_argument("--regrowth-rate", type=float, default=1.45, help="Regrowth rate of the common resource")
parser.add_argument("--cost-multiplier", type=float, default=7, help="Scaling factor for harvest cost")
# rules-based policy-maker parameters
parser.add_argument("--target-health", type=float, default=0.7, help="Target health level for the resource")
parser.add_argument("--target-harvest", type=float, default=0.04, help="Target average harvest level for the agents")
parser.add_argument("--min-reward", type=float, default=1.5, help="If average reward falls below this level, the policymaker will increase taxes")
parser.add_argument("--health-wt", type=float, default=0.6, help="Weight for health deviation in policymaker's tax adjustment")
parser.add_argument("--harvest-wt", type=float, default=0.3, help="Weight for harvest deviation in policymaker's tax adjustment")
parser.add_argument("--reward-wt", type=float, default=0.4, help="Weight for reward deviation in policymaker's tax adjustment")
# llm-based policy-maker parameters
parser.add_argument("--model", type=str, default="gemma3:1b-it-qat", help="LLM model to use for policymaking")
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

if not args.disable_wandb:
    group = "rule-based" if not args.use_llm else "llm-based"
    run = wandb.init(
        project="policy-simulation", name=args.run, group=group, config=vars(args)
    )
else:
    run = None

field_healths = []
for round in range(args.steps):
    tax_rate = policymaker.act(**obs)
    obs, field_health, info = sim.step(tax_rate)
    field_healths.append(field_health)
    log = dict(tax_rate=tax_rate, **obs)
    if run is not None:
        run.log(log)
    log_str = " | ".join(f"{k}: {v:.4f}" for k, v in log.items())
    print(f"| {round=:3} | {log_str} |")

if run is not None:
    final50_healths = field_healths[-50:]
    avg_final50_health = sum(final50_healths) / len(final50_healths)
    run.summary["avg_final50_health"] = avg_final50_health
    run.finish()