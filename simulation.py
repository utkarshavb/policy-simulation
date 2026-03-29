import random

class Agent:
    def __init__(self, greed_level: float=0.0):
        self.g = greed_level
        self.choices = [0.2, 0.8]

    def get_harvest(self, tax_rate, H, redistribution_estimate):
        raise NotImplementedError

class PolicyMaker:
    def __init__(self):
        pass

    def act(self, field_health, avg_harvest, avg_reward):
        raise NotImplementedError

class CommonsSim:
    def __init__(
        self, n_agents: int=10, seed: float=42, regrowth_rate: float=0.2, epsilon: float=0.05
    ):
        self.n_agents = n_agents
        random.seed(seed)
        self.agents = [Agent(random.random()) for _ in range(n_agents)]
        self.H = 1.0
        self.regrowth_rate = regrowth_rate
        self.epsilon = epsilon
        self.redistributions = []
    
    def reset(self):
        self.H = 1.0

    def _get_regrowth(self):
        effective_H = max(self.H, self.epsilon)
        regrowth = self.regrowth_rate * effective_H * (1-self.H)
        return regrowth

    def step(self, tax_rate):
        redistribution_estimate = sum(self.redistributions)/len(self.redistributions)
        harvest_rates = [
            agent.get_harvest(tax_rate, self.H, redistribution_estimate) for agent in self.agents
        ]
        tot_harvest = sum(self.H*a for a in harvest_rates)
        tot_harvest = min(tot_harvest, 1.0)
        self.H -= tot_harvest
        self.H += self._get_regrowth()
        self.H = min(self.H, 1.0)

        redistribution = (tax_rate * tot_harvest)/self.n_agents
        self.redistributions.append(redistribution)

        return self.H