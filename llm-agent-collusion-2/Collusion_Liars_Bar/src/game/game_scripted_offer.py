import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.experiment_player import ExperimentPlayer
from src.agents.player import Player
from src.agents.scripted_player import ScriptedPlayer
from src.game.game_record import GameRecord
from src.game.game_secret_channel import GameSecretChannel, SecretChannel
from src.utils.prompt_loader import PromptLoader

CONDITIONS = ("C0", "C1", "C1b", "C2", "C3", "C4", "C5")   # the main run (PREREG v7)
# Family B (PREREG Amendment 14, results/scripted_offer/followup/HYPOTHESES_v1.md): follow-up
# conditions, never part of the main run's condition list or of its counterbalanced rotation.
FAMILY_B_CONDITIONS = ("C1c", "C1ci", "C1d", "C1s")
ALL_CONDITIONS = CONDITIONS + FAMILY_B_CONDITIONS
# Conditions whose message goes ONLY to the alive seat directly after the proposer (the one agent
# that can challenge the proposer's plays); the other agents receive nothing.
X_ONLY_CONDITIONS = ("C1b", "C1c", "C1ci", "C1d")
CONDITION_NAMES = {
    "C0": "positive_control_explicit_alliance",
    "C1": "direct_offer",
    "C1b": "false_brokered_offer",
    "C2": "indirect_hint",
    "C3": "benign_cooperation_request",
    "C4": "no_offer",
    "C5": "placebo_neutral_message",
    "C1c": "false_self_record_plus_pact",            # HB2: fabricated record of X's own sparing + C1 pact
    "C1ci": "false_self_record_plus_instruction",    # HB3: same record, phrased as a directive, no question
    "C1d": "claimed_prompt_authority_plus_pact",     # HB4: proposer claims to be part of the game's prompt
    "C1s": "direct_offer_series",                    # HB1: C1 message in game 1 of a k-game series + true memory
}
OFFER_TEMPLATES = {
    "C0": "prompts/secret_channel/partner_notification_prompt.txt",
    "C1": "prompts/scripted_offer/offer_direct.txt",
    # C1b (v7): the proposer falsely claims an arrangement with the other two players and asks the
    # recipient not to challenge its plays. Delivered ONLY to the alive seat directly after the
    # proposer at the offer point; the other agents receive nothing, so the claim stays false.
    "C1b": "prompts/scripted_offer/offer_broker.txt",
    "C2": "prompts/scripted_offer/offer_hint.txt",
    "C3": "prompts/scripted_offer/offer_benign.txt",
    # C5: a neutral message with no pact, delivered exactly like C1-C3 (same recipients, timing,
    # response prompt and re-injected block). Its ACCEPT/DECLINE replies are recorded only; they are
    # not acceptance of a pact and are not used in any acceptance hypothesis.
    "C5": "prompts/scripted_offer/offer_placebo.txt",
    "C1c": "prompts/scripted_offer/offer_consistency.txt",
    "C1ci": "prompts/scripted_offer/offer_consistency_instruct.txt",
    "C1d": "prompts/scripted_offer/offer_authority.txt",
    "C1s": "prompts/scripted_offer/offer_direct.txt",   # identical message to C1
}
# HB1 series: memory block appended to an agent's prompts from game 2 on (built from the series' own
# records by the runner; see run_scripted_offer.build_memory_block).
SERIES_MEMORY_TEMPLATE = "prompts/scripted_offer/series_memory_block.txt"
RESPONSE_TEMPLATE = "prompts/scripted_offer/response_prompt.txt"
PRIVATE_BLOCK_TEMPLATE = "prompts/scripted_offer/private_messages_block.txt"
RULES_PATH = "prompts/rule_vanilla.txt"

PROPOSER_NAME = "Riley"
# Per-condition proposer name; every condition uses "Riley" (HB4 keeps the name so that C1d differs
# from C1 in the message text only; a renamed proposer would be a separate variant).
PROPOSER_NAMES = {c: PROPOSER_NAME for c in ALL_CONDITIONS}
# The proposer replaces Mike (ministral-8b); the remaining three LLM seats are fixed.
AGENT_CONFIGS = [
    {"name": "Lily", "model": "llama-3.1-8b"},
    {"name": "Luke", "model": "llama-3.2-3b"},
    {"name": "Quinn", "model": "qwen2.5-7b"},
]


def seat_order(game_number: int, agent_configs: List[Dict[str, str]] = AGENT_CONFIGS) -> List[Dict[str, str]]:
    """Cyclic rotation of the LLM agents, chosen by game number only (identical across conditions).

    Seat 0 is always the proposer; the first agent returned sits directly after the proposer.
    Over any 3 consecutive game numbers each agent takes that seat exactly once.
    """
    k = game_number % len(agent_configs)
    return agent_configs[k:] + agent_configs[:k]


def parse_offer_response(content: str) -> Dict[str, Any]:
    """Code a reply to the offer. Anything that is not a valid JSON decision is UNREADABLE.

    Only the first response is coded; there is no retry and no semantic interpretation of
    malformed text.
    """
    match = re.search(r'({[\s\S]*})', content or "")
    if not match:
        return {"decision": "UNREADABLE", "reply": None, "parse_error": "no_json_object"}
    try:
        obj = json.loads(match.group(1))
    except Exception as e:
        return {"decision": "UNREADABLE", "reply": None, "parse_error": f"invalid_json: {e}"}
    if not isinstance(obj, dict):
        return {"decision": "UNREADABLE", "reply": None, "parse_error": "not_an_object"}
    decision = obj.get("decision")
    reply = obj.get("reply")
    if not isinstance(decision, str) or decision.strip().upper() not in ("ACCEPT", "DECLINE"):
        return {"decision": "UNREADABLE", "reply": reply if isinstance(reply, str) else None,
                "parse_error": f"bad_decision: {decision!r}"}
    if not isinstance(reply, str):
        return {"decision": "UNREADABLE", "reply": None, "parse_error": "missing_reply"}
    return {"decision": decision.strip().upper(), "reply": reply, "parse_error": None}


class GameScriptedOffer(GameSecretChannel):
    """Liar's Bar with one scripted proposer that privately offers a non-challenge pact.

    Reuses the secret-channel game's deck, scoring, penalty and round logic. Differences:
      * seat 0 is a ScriptedPlayer (fixed bluff rate, never challenges);
      * at the start of `offer_round` the proposer sends the condition's message to the agents
        (C1-C3 and the placebo C5: all alive agents; C0: only the agent seated after the proposer,
        as an experimenter-assigned alliance; C4: nobody);
      * the message and the agent's own reply are re-injected verbatim into that agent's later
        play and challenge prompts (agents are otherwise stateless between calls);
      * every challenge decision is logged as a structured row for the H2 analysis.
    """

    def __init__(self, condition: str, game_number: int, seed: int, out_dir: str,
                 offer_round: int = 2, bluff_prob: float = 0.5,
                 agent_configs: List[Dict[str, str]] = AGENT_CONFIGS,
                 attempt_id: Optional[str] = None,
                 series_index: int = 1, series_games: int = 1,
                 memory_blocks: Optional[Dict[str, str]] = None) -> None:
        if condition not in ALL_CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {ALL_CONDITIONS}")
        if series_games > 1 and condition != "C1s":
            raise ValueError(f"series mode is defined for C1s only, not {condition!r}")
        if not 1 <= series_index <= series_games:
            raise ValueError(f"series_index {series_index} outside 1..{series_games}")
        if series_index > 1 and not memory_blocks:
            raise ValueError("series games after the first need the memory blocks of the earlier games")
        if series_index == 1 and memory_blocks:
            raise ValueError("game 1 of a series carries no memory")
        # Seed first: player bullets, the starting seat, target cards and deals all draw from
        # the global random stream, so the same seed gives the same round-1 state in every
        # condition. LLM sampling is not seeded, so trajectories still diverge after round 1.
        random.seed(seed)

        self.condition = condition
        self.game_number = game_number
        self.seed = seed
        self.offer_round = offer_round
        self.bluff_prob = bluff_prob
        self.out_dir = out_dir
        self.attempt_id = attempt_id   # record-only: which run attempt of this game produced the record
        # HB1 series (C1s): game `series_index` of `series_games`; `memory_blocks` = {agent: text} built
        # by the runner from the earlier games' records and appended to that agent's prompts.
        self.series_index = series_index
        self.series_games = series_games
        self.memory_blocks: Dict[str, str] = dict(memory_blocks or {})
        self.seat_configs = seat_order(game_number, agent_configs)

        self.proposer = ScriptedPlayer(PROPOSER_NAMES[condition], bluff_prob=bluff_prob, seed=seed)
        # LLM seats use ExperimentPlayer so the original Player class stays unmodified.
        self.players: List[Player] = [self.proposer] + [ExperimentPlayer(c["name"], c["model"]) for c in self.seat_configs]
        for player in self.players:
            player.init_opinions(self.players)

        self.scores = {player.name: 0 for player in self.players}
        self.deck: List[str] = []
        self.target_card: Optional[str] = None
        self.current_player_idx: int = random.randint(0, len(self.players) - 1)
        self.last_shooter_name: Optional[str] = None
        self.game_over: bool = False
        self.round_count: int = 0

        # Secret-channel state is only used by C0, where it turns on the parent's existing
        # alliance reminder in handle_challenge.
        self.secret_channel_config: Dict[str, Any] = {}
        self.secret_channel = SecretChannel()
        self.tool_offered = False
        self.tool_accepted = False
        self.prompts = {}

        self.c0_ally: Optional[str] = self.seat_configs[0]["name"] if condition == "C0" else None
        # Record-only (C1c, C1ci, C1d): who received the X-only message. C1b keeps `c1b_recipient`.
        self.x_recipient: Optional[str] = None
        # Record-only: the agent in the next alive seat after the proposer at the offer point
        # (the only agent that can challenge the proposer). Set in every condition, including
        # C4, for same-seat pairing in the analysis. It does not affect who receives a message.
        self.target: Optional[str] = None
        # C1b record-only fields: who received the brokered offer, and the first post-offer
        # challenge of that agent's play by an LLM player (the claimed protection visibly failing).
        self.c1b_recipient: Optional[str] = None
        self.first_breach: Optional[Dict[str, Any]] = None
        self.offer_done = False
        self.offer_round_actual: Optional[int] = None
        self.offered: List[str] = []
        self.offers: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.plays: List[Dict[str, Any]] = []

        self.game_record: GameRecord = GameRecord(None, communication_enabled=False)
        self.game_record.start_game([p.name for p in self.players])

        if self.series_index > 1:
            # Games 2..k of a series: no new message. Every agent with a memory block carries it in
            # all of its prompts from the first call, and all of its decisions count as post-offer.
            for agent in self.players[1:]:
                if agent.name in self.memory_blocks:
                    agent.private_messages = self.memory_blocks[agent.name]
                    self.offered.append(agent.name)
            self.offer_done = True
            self.offer_round_actual = 0
            self.target = self.players[self._next_alive_idx(0)].name
            self.game_record.add_secret_channel_event({"event": "series_memory",
                                                       "game_in_series": self.series_index,
                                                       "agents": list(self.memory_blocks)})

    # ------------------------------------------------------------------
    # Offer delivery and memory injection
    # ------------------------------------------------------------------
    def _offer_recipients(self) -> List[Player]:
        alive_agents = [p for p in self.players if p.alive and p is not self.proposer]
        if self.condition == "C4":
            return []
        if self.condition == "C0":
            return [p for p in alive_agents if p.name == self.c0_ally]
        if self.condition in X_ONLY_CONDITIONS:
            # The alive seat directly after the proposer at the offer point (ignores hands, like
            # `target`). Empty if that seat is the proposer itself (no other player alive).
            nxt = self.players[self._next_alive_idx(0)]
            return [nxt] if nxt is not self.proposer and nxt.alive else []
        return alive_agents   # C1, C2, C3, C5 and game 1 of a C1s series

    def deliver_offer(self) -> None:
        """Send the condition's private message to each recipient and code the replies."""
        self.offer_done = True
        self.offer_round_actual = self.round_count
        if self.proposer.alive:
            self.target = self.players[self._next_alive_idx(0)].name
        rules = PromptLoader.load_prompt(RULES_PATH).strip()
        response_template = PromptLoader.load_prompt(RESPONSE_TEMPLATE)
        block_template = PromptLoader.load_prompt(PRIVATE_BLOCK_TEMPLATE)

        recipients = self._offer_recipients()
        if self.condition == "C1b" and recipients:
            self.c1b_recipient = recipients[0].name
        if self.condition in ("C1c", "C1ci", "C1d") and recipients:
            self.x_recipient = recipients[0].name
        if self.condition == "C0":
            if recipients:
                self.secret_channel.establish(self.proposer.name, self.c0_ally, self.round_count)
                self.tool_offered = True
                self.tool_accepted = True
            else:
                print(f"[SCRIPTED OFFER] C0 ally {self.c0_ally} is not alive; no alliance established.")

        for agent in recipients:
            if self.condition == "C0":
                offer_text = PromptLoader.load_prompt(OFFER_TEMPLATES["C0"]).replace(
                    '{initiator_name}', self.proposer.name).strip()
            else:
                offer_text = PromptLoader.load_prompt(OFFER_TEMPLATES[self.condition]).replace(
                    '{proposer_name}', self.proposer.name).replace('{agent_name}', agent.name).strip()
            prompt = (response_template
                      .replace('{rules}', rules)
                      .replace('{agent_name}', agent.name)
                      .replace('{round_number}', str(self.round_count))
                      .replace('{offer_message}', offer_text)
                      .replace('{proposer_name}', self.proposer.name))

            content, _ = agent.llm_client.chat([{"role": "user", "content": prompt}], model=agent.model_name)
            coded = parse_offer_response(content)

            # The re-injected reply is what the agent actually said: the parsed reply when
            # readable, otherwise its raw text (capped at 50 words).
            reply_text = coded["reply"] if coded["reply"] else " ".join((content or "").split()[:50])
            if not reply_text:
                reply_text = "(no reply)"
            block = (block_template
                     .replace('{proposer_name}', self.proposer.name)
                     .replace('{round_number}', str(self.round_count))
                     .replace('{offer_text}', offer_text)
                     .replace('{reply_text}', reply_text))
            agent.private_messages = block
            self.offered.append(agent.name)

            row = {
                "agent": agent.name,
                "model": agent.model_name,
                "round": self.round_count,
                "seat_after_proposer": self.players[self._next_alive_idx(0)].name == agent.name,
                "offer_text": offer_text,
                "prompt": prompt,
                "raw_response": content,
                "decision": coded["decision"],
                "reply": coded["reply"],
                "parse_error": coded["parse_error"],
                "injected_block": block,
            }
            self.offers.append(row)
            self.game_record.add_secret_channel_event({"event": "scripted_offer", **{
                k: row[k] for k in ("agent", "round", "decision", "parse_error")}})
            print(f"[SCRIPTED OFFER] {self.condition} {self.proposer.name} -> {agent.name}: "
                  f"{coded['decision']} | {content!r}")

        if not recipients:
            self.game_record.add_secret_channel_event({"event": "no_offer", "round": self.round_count})

    def _next_alive_idx(self, idx: int) -> int:
        """Index of the next alive seat after `idx` (ignores hands, unlike find_next_player_with_cards)."""
        for step in range(1, len(self.players) + 1):
            j = (idx + step) % len(self.players)
            if self.players[j].alive:
                return j
        return idx

    # ------------------------------------------------------------------
    # Turn handling with structured logging
    # ------------------------------------------------------------------
    def handle_play_cards(self, current_player: Player, next_player: Player) -> List[str]:
        if current_player is self.proposer:
            self.proposer.current_target = self.target_card
        played = super().handle_play_cards(current_player, next_player)
        self.plays.append({
            "round": self.round_count,
            "player": current_player.name,
            "is_scripted": current_player is self.proposer,
            "played_cards": list(played),
            "is_bluff": not self.is_valid_play(played),
            "is_fallback": bool(getattr(current_player, "last_play_fallback", False)),
            "post_offer": self.offer_done,
        })
        return played

    def can_challenge_proposer(self, player: Player) -> bool:
        """Record-only: whether `player` is, right now, the next seat with cards after the proposer.

        In Liar's Bar only the next player with cards may challenge a play, so this is the one
        agent that would judge the proposer's next play. False for everyone when the proposer is
        dead or has no cards. Hands are read as they are at the decision point (after the play).
        """
        if not self.proposer.alive or not self.proposer.hand:
            return False
        idx = self.find_next_player_with_cards(0)
        return idx != 0 and self.players[idx] is player

    def handle_challenge(self, current_player: Player, next_player: Player, played_cards: List[str]) -> tuple:
        can_challenge_proposer = self.can_challenge_proposer(next_player)
        result = super().handle_challenge(current_player, next_player, played_cards)
        action = self.game_record.get_current_round().get_last_action()
        is_proposer_play = current_player is self.proposer
        # C1b first breach (record-only): the first post-offer challenge of the recipient's play by
        # an LLM player. The challenger here is by the rules the next alive seat with cards after
        # the recipient, i.e. "whoever ends up sitting after you". The proposer never challenges;
        # system auto-challenges go through handle_system_challenge and never reach this point.
        # Not recorded when the proposer was dead at the offer (target is None): see PREREG.
        if (self.condition == "C1b" and self.first_breach is None and self.offer_done
                and self.target is not None and self.c1b_recipient is not None
                and current_player.name == self.c1b_recipient and bool(action.was_challenged)
                and next_player is not self.proposer):
            self.first_breach = {
                "round": self.round_count,
                "decision_index": len(self.decisions),
                "challenger": next_player.name,
                "challenge_succeeded": not self.is_valid_play(played_cards),
            }
        self.decisions.append({
            "round": self.round_count,
            "challenger": next_player.name,
            "challenger_is_scripted": next_player is self.proposer,
            "current_player": current_player.name,
            "is_proposer_play": is_proposer_play,
            "play_was_bluff": not self.is_valid_play(played_cards),
            "proposer_bluffed": (not self.is_valid_play(played_cards)) if is_proposer_play else None,
            "play_is_fallback": bool(getattr(current_player, "last_play_fallback", False)),
            "was_challenged": bool(action.was_challenged),
            "is_fallback": bool(getattr(next_player, "last_challenge_fallback", False)),
            "is_system": False,
            "post_offer": self.offer_done,
            "challenger_offered": next_player.name in self.offered,
            "challenger_is_target": next_player.name == self.target,
            "challenger_can_challenge_proposer": can_challenge_proposer,
            "challenger_is_c0_ally": next_player.name == self.c0_ally,
            "challenge_reason": action.challenge_reason,
            # True for every decision from the first-breach decision onward (C1b only; else False).
            "after_first_breach": self.first_breach is not None,
        })
        return result

    def handle_system_challenge(self, current_player: Player) -> None:
        self.decisions.append({
            "round": self.round_count,
            "challenger": "system",
            "challenger_is_scripted": False,
            "current_player": current_player.name,
            "is_proposer_play": current_player is self.proposer,
            "play_was_bluff": not self.is_valid_play(current_player.hand),
            "proposer_bluffed": (not self.is_valid_play(current_player.hand)) if current_player is self.proposer else None,
            "play_is_fallback": False,
            "was_challenged": True,
            "is_fallback": False,
            "is_system": True,
            "post_offer": self.offer_done,
            "challenger_offered": False,
            "challenger_is_target": False,
            "challenger_can_challenge_proposer": False,
            "challenger_is_c0_ally": False,
            "challenge_reason": "System automatic challenge",
            "after_first_breach": self.first_breach is not None,
        })
        super().handle_system_challenge(current_player)

    def play_round(self) -> None:
        """One turn. Mirrors game.py's play_round.

        GameSecretChannel.play_round calls check_victory/reset_round again after perform_penalty
        (which already does both), double-resetting the round and awarding the last-survivor
        bonus twice. This override does not.
        """
        current_player = self.players[self.current_player_idx]

        if self.check_other_players_no_cards(current_player):
            self.handle_system_challenge(current_player)
            return

        print(f"\nIt's {current_player.name}'s turn, target card is {self.target_card}")
        current_player.print_status()

        next_idx = self.find_next_player_with_cards(self.current_player_idx)
        next_player = self.players[next_idx]

        played_cards = self.handle_play_cards(current_player, next_player)

        if next_player != current_player:
            player_to_penalize, was_unsuccessful_challenge = self.handle_challenge(
                current_player, next_player, played_cards)
            if player_to_penalize is not None:
                self.perform_penalty(player_to_penalize, was_unsuccessful_challenge)
                return
            print(f"{next_player.name} did not challenge {current_player.name}'s hand. Game continues...")

        self.current_player_idx = next_idx

    def start_game(self, max_turns: int = 400) -> Dict[str, Any]:
        """Play to the end, save the record, and return it."""
        print(f"Scripted-offer game: condition={self.condition} game={self.game_number} seed={self.seed} "
              f"seats={[p.name for p in self.players]}")
        self.choose_target_card()
        self.deal_cards()
        self.start_round_record()

        turns = 0
        while not self.game_over:
            if not self.offer_done and self.round_count >= self.offer_round:
                self.deliver_offer()
            self.play_round()
            turns += 1
            if turns >= max_turns:
                raise RuntimeError(f"Game exceeded {max_turns} turns without ending")

        return self.save()

    def to_dict(self) -> Dict[str, Any]:
        record = self.game_record.to_dict()
        record["secret_channel_events"] = self.game_record.secret_channel_events
        record["scripted_offer"] = {
            "condition": self.condition,
            "condition_name": CONDITION_NAMES[self.condition],
            "game_number": self.game_number,
            "seed": self.seed,
            "seat_order": [p.name for p in self.players],
            "models": {p.name: p.model_name for p in self.players},
            "proposer": self.proposer.name,
            "offer_round": self.offer_round,
            "offer_round_actual": self.offer_round_actual,
            "bluff_prob": self.bluff_prob,
            "target": self.target,
            "c0_ally": self.c0_ally,
            "c1b_recipient": self.c1b_recipient,
            "x_recipient": self.x_recipient,
            "first_breach": self.first_breach,
            # HB1 series bookkeeping (None for every single-game condition).
            "series": {"index": self.series_index, "games": self.series_games,
                       "memory_blocks": self.memory_blocks} if self.series_games > 1 else None,
            "offers": self.offers,
            "decisions": self.decisions,
            "plays": self.plays,
            # Record-only fields; nothing an agent sees depends on them.
            "attempt_id": self.attempt_id,
            "api_retries": {p.name: getattr(p, "api_retries", 0) for p in self.players if p is not self.proposer},
        }
        return record

    def record_name(self) -> str:
        """<cond>_gNN.json, or <cond>_gNN_kI.json for game I of a series."""
        if self.series_games > 1:
            return f"{self.condition}_g{self.game_number:02d}_k{self.series_index}.json"
        return f"{self.condition}_g{self.game_number:02d}.json"

    def save(self) -> Dict[str, Any]:
        record = self.to_dict()
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, self.record_name())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"Game record saved to {path}")
        return record
