import random
import math

class Agent:
    def __init__(self, greed_level: float=0.0):
        self.greed = greed_level
        self.choices = [0.2, 0.8]

    def get_harvest(self, tax_rate: float, H: float, rng: random.Random):
        expected_rewards = [(1-tax_rate)*a*H for a in self.choices]
        wts = [math.exp(r*self.greed) for r in expected_rewards]
        harvest_rate = rng.choices(self.choices, weights=wts, k=1)[0]
        harvest = harvest_rate * H
        return harvest

class CommonsSim:
    def __init__(
        self, n_agents: int=10, seed: float=42, regrowth_rate: float=0.2, epsilon: float=0.05
    ):
        self.n_agents = n_agents
        self.seed = seed
        self.rng = random.Random(seed)
        self.agents = [Agent(self.rng.random()) for _ in range(n_agents)]
        self.H = 1.0
        self.regrowth_rate = regrowth_rate
        self.epsilon = epsilon
    
    def reset(self):
        self.H = 1.0
        self.rng = random.Random(self.seed)
        self.agents = [Agent(self.rng.random()) for _ in range(self.n_agents)]
        obs = dict(field_health=self.H,avg_harvest=0.0, avg_reward=0.0)
        return obs

    def _get_regrowth(self):
        effective_H = max(self.H, self.epsilon)
        regrowth = self.regrowth_rate * effective_H * (1-self.H)
        return regrowth

    def step(self, tax_rate: float):
        individual_harvests = [
            agent.get_harvest(tax_rate, self.H, rng=self.rng) for agent in self.agents
        ]
        tot_harvest = sum(individual_harvests)
        if tot_harvest > self.H:
            individual_harvests = [h*(self.H/tot_harvest) for h in individual_harvests]
            tot_harvest = self.H
        
        taxes = [tax_rate*h for h in individual_harvests]
        redistribution = sum(taxes) / self.n_agents
        rewards = [(h-t)+redistribution for h, t in zip(individual_harvests, taxes)]

        self.H -= tot_harvest
        self.H += self._get_regrowth()
        self.H = min(self.H, 1.0)
        if self.H < 0.3:
            rewards = [0.5*r for r in rewards]

        obs = dict(
            field_health=self.H,
            avg_harvest=sum(individual_harvests)/self.n_agents,
            avg_reward=sum(rewards)/self.n_agents
        )
        info = dict(
            individual_harvests=individual_harvests, rewards=rewards,
            redistribution=redistribution,
        )
        return obs, self.H, info