class RuleBasedPolicymaker:
    def __init__(
        self, target_health=0.5, target_harvest=0.045, min_reward=0.08,
        health_wt=0.6, harvest_wt=0.3, reward_wt=0.4
    ):
        self.target_health = target_health
        self.target_harvest = target_harvest
        self.min_reward = min_reward
        self.health_wt = health_wt
        self.harvest_wt = harvest_wt
        self.reward_wt = reward_wt
        self.tax = 0.3

    def act(self, field_health, avg_harvest, avg_reward):
        delta = (
            self.health_wt * (self.target_health - field_health)
            + self.harvest_wt * (avg_harvest - self.target_harvest)
            - self.reward_wt * max(0.0, self.min_reward - avg_reward)
        )
        self.tax = max(0.0, min(1.0, self.tax + delta))
        return self.tax