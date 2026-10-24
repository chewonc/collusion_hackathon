import random
from typing import Dict, List, Optional

from src.agents.player import Player


class ScriptedPlayer(Player):
    """A rule-based seat that never calls an LLM.

    Used as the pact proposer in the scripted-offer experiment. Its behaviour is identical in
    every condition: it plays exactly one card per turn, bluffs with a fixed probability, and
    never challenges anyone. Only the private message it sends differs between conditions.
    """

    BEHAVIOR = "Places one card face down without comment."

    def __init__(self, name: str, bluff_prob: float = 0.5, seed: int = 0):
        # Deliberately skip Player.__init__ so no LLM client is created. The bullet position is
        # drawn from the global `random` module, exactly as Player does, so the global random
        # stream (and therefore deals and bullets) is consumed the same way as with an LLM seat.
        self.name = name
        self.hand: List[str] = []
        self.alive = True
        self.bullet_position = random.randint(0, 5)
        self.current_bullet_position = 1
        self.opinions: Dict[str, str] = {}
        self.series_memory = ""
        self.private_messages = ""
        self.secret_channel_memory = {
            'enabled': False,
            'partner_name': None,
            'communication_history': [],
            'game_history': [],
            'alliance_established_game': None
        }
        self.llm_client = None
        self.model_name = "scripted"
        self.is_scripted = True
        self.last_play_fallback = False
        self.last_challenge_fallback = False

        self.bluff_prob = bluff_prob
        # Private RNG so the proposer's bluffing never consumes the global random stream.
        self.rng = random.Random(f"proposer-{seed}")
        # Set by the game before each play; needed to tell honest cards from bluff cards.
        self.current_target: Optional[str] = None
        self.last_play_bluffed: Optional[bool] = None

    def choose_cards_to_play(self, round_base_info: str, round_action_info: str,
                             play_decision_info: str):
        """Play one card: a non-target card with probability `bluff_prob`, otherwise an honest one."""
        target = self.current_target
        honest = [c for c in self.hand if c == target or c == 'Joker']
        bluff = [c for c in self.hand if c != target and c != 'Joker']

        want_bluff = self.rng.random() < self.bluff_prob
        if want_bluff and bluff:
            pool = bluff
        elif honest:
            pool = honest
        else:
            pool = bluff
        card = self.rng.choice(sorted(pool))
        self.hand.remove(card)
        self.last_play_bluffed = card in bluff

        result = {
            "played_cards": [card],
            "behavior": self.BEHAVIOR,
            "play_reason": "scripted",
        }
        return result, ""

    def decide_challenge(self, round_base_info: str, round_action_info: str,
                         challenge_decision_info: str, challenging_player_performance: str,
                         extra_hint: str):
        """Never challenge, in every condition."""
        return {"was_challenged": False, "challenge_reason": "scripted", "is_fallback": False}, ""

    def reflect(self, alive_players: List[str], round_base_info: str, round_action_info: str,
                round_result: str) -> None:
        """No-op: the scripted seat keeps no impressions."""
        return None
