"""Deterministic offline stand-in for LLMClient (no network).

Every answer is a pure function of the prompt text, so identical game states produce identical
answers and whole games are reproducible. A fixed share of answers is malformed on purpose so
the fallback and UNREADABLE paths get exercised.
"""
import hashlib
import json
import re
from typing import Dict, List, Tuple


class MockLLMClient:
    # Every prompt sent through any MockLLMClient, in order: (model, prompt text).
    calls: List[Tuple[str, str]] = []

    def __init__(self, malformed_rate: float = 0.1):
        self.malformed_rate = malformed_rate

    @staticmethod
    def _u(prompt: str, salt: str) -> float:
        h = hashlib.sha256((salt + prompt).encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def chat(self, messages: List[Dict[str, str]], model: str = "mock",
             max_new_tokens: int = 4096) -> Tuple[str, str]:
        prompt = messages[-1]["content"]
        MockLLMClient.calls.append((model, prompt))
        if self._u(prompt, "malformed") < self.malformed_rate:
            return "I am not sure what to do here.", ""

        if '"decision"' in prompt:
            u = self._u(prompt, "decision")
            decision = "ACCEPT" if u < 0.6 else "DECLINE"
            return json.dumps({"decision": decision, "reply": f"Mock reply: {decision.lower()}."}), ""

        if '"was_challenged"' in prompt:
            challenge = self._u(prompt, "challenge") < 0.4
            return json.dumps({"was_challenged": challenge, "challenge_reason": "mock"}), ""

        if '"played_cards"' in prompt:
            hands = re.findall(r"Your current hand is: ([^\n]+)", prompt)
            hand = [c.strip() for c in hands[-1].split(",")] if hands else []
            n = 1 + int(self._u(prompt, "count") * 2)
            cards = hand[:max(1, min(n, len(hand)))]
            return json.dumps({"played_cards": cards, "behavior": "mock", "play_reason": "mock"}), ""

        return "Mock impression.", ""
