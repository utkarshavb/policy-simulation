class RuleBasedPolicymaker:
    def __init__(
        self, target_health=0.7, target_harvest=0.45, reward_floor=0.08, k_h=0.6, k_a=0.3, k_r=0.4
    ):
        self.target_health = target_health
        self.target_harvest = target_harvest
        self.reward_floor = reward_floor
        self.k_h = k_h
        self.k_a = k_a
        self.k_r = k_r
        self.tax = 0.3

    def act(self, field_health, avg_harvest, avg_reward):
        delta = (
            self.k_h * (self.target_health - field_health)
            + self.k_a * (avg_harvest - self.target_harvest)
            - self.k_r * max(0.0, self.reward_floor - avg_reward)
        )

        # delta = max(-max_step, min(max_step, delta))
        self.tax = max(0.0, min(1.0, self.tax + delta))
        return self.tax