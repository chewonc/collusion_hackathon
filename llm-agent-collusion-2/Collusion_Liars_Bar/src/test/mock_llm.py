"""Deterministic offline stand-in for LLMClient (no network).

Every answer is a pure function of the prompt text, so identical game states produce identical
answers and whole games are reproducible. A fixed share of answers is malformed on purpose so
the fallback and UNREADABLE paths get exercised.
"""
import hashlib
import os
import time
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


# ----------------------------------------------------------------------
# Fake OpenAI transport with fault injection (offline; for the API-error tests)
# ----------------------------------------------------------------------
SITES = ("play", "challenge", "offer", "reflect")


def prompt_site(prompt: str) -> str:
    """Which model call a prompt belongs to, read from the JSON keys it asks for."""
    if '"played_cards"' in prompt:
        return "play"
    if '"was_challenged"' in prompt:
        return "challenge"
    if '"decision"' in prompt:
        return "offer"
    return "reflect"


class MockFault:
    """A fault for one game, parsed from 'COND:G:SITE:KIND[:N[:SKIP]]'.

    SITE: play | challenge | offer | reflect | any.
    KIND: transient (connection error), fatal (HTTP 401) or hang (sleep N seconds, then answer).
    N: transient/fatal -> number of failing calls (default: every call); hang -> seconds (default 30).
    SKIP: matching calls let through before the fault starts (default 0).
    """

    def __init__(self, spec: str):
        parts = spec.split(":")
        if len(parts) < 4:
            raise ValueError(f"bad --mock-fault {spec!r}; expected COND:G:SITE:KIND[:N[:SKIP]]")
        self.condition, self.game_number, self.site, self.kind = parts[0], int(parts[1]), parts[2], parts[3]
        if self.site not in SITES + ("any",) or self.kind not in ("transient", "fatal", "hang"):
            raise ValueError(f"bad --mock-fault {spec!r}")
        default_n = 30 if self.kind == "hang" else 10 ** 9
        self.n = int(parts[4]) if len(parts) > 4 and parts[4] else default_n
        self.skip = int(parts[5]) if len(parts) > 5 else 0

    def applies(self, condition: str, game_number: int) -> bool:
        return (condition, game_number) == (self.condition, self.game_number)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class MockTransport:
    """Stands in for the OpenAI client inside StrictLLMClient: chat.completions.create(**kwargs).

    Answers come from MockLLMClient; matching calls raise the configured openai exception (or
    hang). One instance per player, so counts are per seat; the fault state is shared through
    the MockFault object so the n-th matching call is counted across seats.
    """

    def __init__(self, fault: MockFault, hang_marker: str = None):
        self.fault = fault
        self.hang_marker = hang_marker
        self.chat = _Obj(completions=_Obj(create=self._create))
        if not hasattr(fault, "seen"):
            fault.seen = 0
            fault.failed = 0

    def _create(self, model=None, messages=None, **kwargs):
        import openai
        import httpx2
        prompt = messages[-1]["content"]
        f = self.fault
        if f.site in ("any", prompt_site(prompt)):
            f.seen += 1
            if f.seen > f.skip and f.kind == "hang" and f.failed == 0:
                f.failed = 1
                if self.hang_marker:
                    with open(self.hang_marker, "w") as fh:
                        fh.write(str(os.getpid()))
                time.sleep(f.n)
            elif f.seen > f.skip and f.kind != "hang" and f.failed < f.n:
                f.failed += 1
                request = httpx2.Request("POST", "http://mock.invalid/v1/chat/completions")
                if f.kind == "transient":
                    raise openai.APIConnectionError(message="mock connection error", request=request)
                raise openai.AuthenticationError("mock 401", response=httpx2.Response(401, request=request),
                                                 body=None)
        text, _ = MockLLMClient().chat(messages, model=model)
        # LLMClient._clean_response would quote the word after "reply: " and break the JSON; the
        # real models' replies are not at issue here, only keeping mock games readable.
        text = text.replace("Mock reply: ", "Mock reply, ")
        return _Obj(choices=[_Obj(message=_Obj(content=text))], usage=None)
