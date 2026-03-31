import random
import math

class Agent:
    """
    Behaviour:
        - Harvest is scaled by a "revenue scalar", reflecting the "revenue" generated from harvesting
        - Concept of a "cost" is introduced which is convex in harvest amount, scaled by a "cost scalar"
        - Agent softmax chooses between a "gentle" and "high" harvest (20% or 100% of fair share)
        - Greed level acts as the softmax temperature, determining how strongly the agent prefers the higher harvest
    """
    def __init__(self, greed_level: float, revenue_scalar: float=100, cost_scalar: float=700):
        self.greed = greed_level
        self.r = revenue_scalar
        self.c = cost_scalar
        self.choices = [0.2, 1.0]   # proportion of fair share to harvest

    def get_harvest(self, tax_rate: float, fair_share: float, rng: random.Random):
        harvest_choices = [fair_share*a for a in self.choices]
        costs = [self.c*h**2 for h in harvest_choices]
        expected_rewards = [self.r*(1-tax_rate)*h - c for h, c in zip(harvest_choices, costs)]

        # softmax with greed as temperature
        beta = 10 * self.greed   # scale greed for sharper preference
        m = max(expected_rewards)
        wts = [math.exp((r-m)*beta) for r in expected_rewards]
        harvest = rng.choices(harvest_choices, weights=wts, k=1)[0]

        return harvest

class CommonsSim:
    def __init__(
        self, n_agents: int=10, seed: float=42, regrowth_rate: float=0.2,
        revenue_scalar: float=100, cost_scalar: float=700
    ):
        self.n_agents = n_agents
        self.seed = seed
        self.rng = random.Random(seed)
        self.r = revenue_scalar
        self.c = cost_scalar
        self.agents = [Agent(self.rng.random(), self.r, self.c) for _ in range(n_agents)]
        self.H = 1.0
        self.regrowth_rate = regrowth_rate
    
    def reset(self):
        self.H = 1.0
        self.rng = random.Random(self.seed)
        self.agents = [Agent(self.rng.random(), self.r,self.c) for _ in range(self.n_agents)]
        obs = dict(field_health=self.H, avg_harvest=0.0, avg_reward=0.0)
        return obs

    def step(self, tax_rate: float):
        # determine harvests
        fair_share = self.H / self.n_agents
        harvests = [
            agent.get_harvest(tax_rate, fair_share, rng=self.rng) for agent in self.agents
        ]
        tot_harvest = sum(harvests)
        if tot_harvest > self.H:
            harvests = [h*(self.H/tot_harvest) for h in harvests]
            tot_harvest = self.H

        # update resource
        regrowth = self.regrowth_rate * self.H * (1-self.H)
        self.H = self.H - tot_harvest + regrowth
        self.H = min(self.H, 1.0)

        # calculate rewards
        taxes = [self.r*tax_rate*h for h in harvests]
        costs = [self.c*h**2 for h in harvests]
        redistribution = sum(taxes) / self.n_agents
        rewards = [self.r*h-c-t+redistribution for h, c, t in zip(harvests, costs, taxes)]
        if self.H < 0.3:
            rewards = [0.5*r for r in rewards]

        # logging info
        obs = dict(
            field_health=self.H,
            avg_harvest=sum(harvests)/self.n_agents,
            avg_reward=sum(rewards)/self.n_agents
        )
        info = dict(
            harvests=harvests, rewards=rewards, redistribution=redistribution, taxes=taxes
        )
        return obs, self.H, info