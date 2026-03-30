import json
import re
import requests

class RuleBasedPolicymaker:
    def __init__(
        self, target_health=0.5, target_harvest=0.045, min_reward=0.08,
        health_wt=0.6, harvest_wt=0.3, reward_wt=0.4
    ):
        self.target_health, self.health_wt = target_health, health_wt
        self.target_harvest, self.harvest_wt = target_harvest, harvest_wt
        self.min_reward, self.reward_wt = min_reward, reward_wt
        self.tax = 0.3

    def act(self, field_health, avg_harvest, avg_reward):
        delta = (
            self.health_wt * (self.target_health - field_health)
            + self.harvest_wt * (avg_harvest - self.target_harvest)
            - self.reward_wt * max(0.0, self.min_reward - avg_reward)
        )
        self.tax = max(0.0, min(1.0, self.tax + delta))
        return self.tax

class LLMPolicyMaker:
    def __init__(
        self, model: str, base_url: str, prompt_file: str, temperature: float=0.2, timeout: int=60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        with open(prompt_file, "r") as f:
            self.template = f.read()
        self.temperature = temperature
        self.timeout = timeout
        self.field_healths, self.avg_harvests, self.avg_rewards, self.taxes = [], [], [], []

    def _build_prompt(self) -> str:
        rounds = len(self.field_healths)
        data = zip(
            self.taxes[-5:], self.field_healths[-5:], self.avg_harvests[-5:], self.avg_rewards[-5:]
        )
        start = max(1, rounds-4)
        history_text = "\n".join(
            f"Round {i}: tax rate={t:.3f}, field health={h:.3f}, "
            f"avg harvest={ah:.4f}, avg reward={ar:.5f}"
            for i, (t, h, ah, ar) in enumerate(data, start=start)
        )
        prompt = self.template.format(round=rounds+1, history_text=history_text)
        return prompt

    def _query_ollama(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature},
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def _parse_tax(self, text: str) -> float:
        try:
            data = json.loads(text)
            tax = float(data["tax_rate"])
            return max(0.0, min(1.0, tax))
        except Exception:
            # Try extracting JSON object from a noisy response
            match = re.search(r'\{.*?"tax_rate"\s*:\s*([0-9]*\.?[0-9]+).*?\}', text, re.DOTALL)
            if match:
                tax = float(match.group(1))
                return max(0.0, min(1.0, tax))

            # Last-resort: first float in text
            match = re.search(r'([0-9]*\.?[0-9]+)', text)
            if match:
                tax = float(match.group(1))
                return max(0.0, min(1.0, tax))

            # Fallback if model completely fails
            return self.taxes[-1] if self.taxes else 0.3

    def act(self, field_health: float, avg_harvest: float, avg_reward: float):
        # update history
        self.field_healths.append(field_health)
        self.avg_harvests.append(avg_harvest)
        self.avg_rewards.append(avg_reward)

        prompt = self._build_prompt()
        raw = self._query_ollama(prompt)
        tax = self._parse_tax(raw)
        self.taxes.append(tax)
        return tax