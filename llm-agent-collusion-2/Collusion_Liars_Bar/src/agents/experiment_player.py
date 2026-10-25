import json
import re
import time
from collections import Counter
from typing import Callable, Dict, List, Optional

from src.agents.player import (Player, RULE_BASE_PATH, PLAY_CARD_PROMPT_TEMPLATE_PATH,
                               CHALLENGE_PROMPT_TEMPLATE_PATH, REFLECT_PROMPT_TEMPLATE_PATH)
from src.llm import llm_client as llm_client_module
from src.llm.llm_client import LLMClient


# ----------------------------------------------------------------------
# API errors: never turned into a game action
# ----------------------------------------------------------------------
class ApiRequestError(Exception):
    """An inference request failed (as opposed to the model answering badly).

    It is never caught by the players' format-retry loops, so it can never become a fallback
    decision: it ends the game, the runner logs the game as status=error, and the game is rerun.
    """

    def __init__(self, message: str, model: Optional[str] = None, retries: int = 0):
        super().__init__(message)
        self.model = model
        self.retries = retries


class ApiTransientExhausted(ApiRequestError):
    """A transient error (timeout, connection, 408/409/429, 5xx) persisted through the whole backoff."""


class ApiFatalError(ApiRequestError):
    """A non-transient error (authentication, 400, 404, unsupported model, malformed response, ...)."""


_TRANSIENT_STATUS = {408, 409, 429}


def is_transient(exc: BaseException) -> bool:
    import openai
    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return True   # APITimeoutError is a subclass of APIConnectionError
    status = getattr(exc, "status_code", None) if isinstance(exc, openai.APIStatusError) else None
    return status is not None and (status in _TRANSIENT_STATUS or status >= 500)


# Receives one dict per API error (retry or raise). The runner sets it to write api_errors.jsonl.
_api_event_sink: Optional[Callable[[Dict], None]] = None


def set_api_event_sink(sink: Optional[Callable[[Dict], None]]) -> None:
    global _api_event_sink
    _api_event_sink = sink


class StrictLLMClient(LLMClient):
    """LLMClient whose requests are identical but whose API errors are never swallowed.

    LLMClient.chat catches transient errors and returns "", which the players then count as an
    unparseable answer and, after five, turn into a fallback ("do not challenge" / "play one
    card"). Here:
      * SDK-internal retries are off (max_retries=0), so every extra HTTP request is one of ours;
        api_retries counts exactly those requests;
      * transient errors are retried with exponential backoff (BACKOFF_S, about 10 minutes in
        total) and, if they persist, raise ApiTransientExhausted;
      * every other failure raises ApiFatalError at once;
      * every error is reported to the event sink (one event per retry and one per raise).
    Sampling parameters and response clean-up are LLMClient's own (_sampling, _clean_response).
    """

    BACKOFF_S = (2, 4, 8, 16, 32) + (60,) * 9

    def __init__(self, client=None, sleep: Callable[[float], None] = time.sleep):
        self._client = client if client is not None else llm_client_module.make_client(max_retries=0)
        self._sleep = sleep
        self.api_retries = 0
        self.owner: Optional[str] = None
        print(f"StrictLLMClient initialized. Models: {list(self.MODEL_MAP)}")

    def _emit(self, model, exc, transient, retry, wait_s, action):
        if _api_event_sink is not None:
            _api_event_sink({
                "player": self.owner, "model": model, "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None), "transient": transient,
                "retry": retry, "wait_s": wait_s, "action": action, "message": str(exc)[:500],
            })

    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.1-8b",
             max_new_tokens: int = 4096):
        if model not in self.MODEL_MAP:
            exc = ValueError(f"Unsupported model '{model}'. Supported: {list(self.MODEL_MAP)}.")
            self._emit(model, exc, False, 0, None, "raise")
            raise ApiFatalError(str(exc), model=model) from exc
        print(f"LLM Request to {model}: {messages}")
        params = self._sampling(model)
        retries = 0
        while True:
            try:
                completion = self._client.chat.completions.create(
                    model=llm_client_module.resolve_model(self.MODEL_MAP[model]),
                    messages=messages,
                    max_tokens=min(max_new_tokens, params["max_tokens"]),
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    extra_body={"top_k": params["top_k"], "repetition_penalty": params["repetition_penalty"]},
                )
                content = completion.choices[0].message.content or ""
            except Exception as e:
                transient = is_transient(e)
                if transient and retries < len(self.BACKOFF_S):
                    wait = self.BACKOFF_S[retries]
                    retries += 1
                    self.api_retries += 1
                    self._emit(model, e, True, retries, wait, "retry")
                    print(f"LLM API error ({type(e).__name__}), retry {retries} in {wait}s: {e}")
                    self._sleep(wait)
                    continue
                self._emit(model, e, transient, retries, None, "raise")
                cls = ApiTransientExhausted if transient else ApiFatalError
                print(f"LLM API error ({type(e).__name__}), giving up after {retries} retries: {e}")
                raise cls(f"{type(e).__name__}: {e}", model=model, retries=retries) from e
            response = self._clean_response(content, model)
            print(f"LLM Response: {response}")
            return response, ""


# Tests can make every ExperimentPlayer use a given client (e.g. a StrictLLMClient over a fake
# transport). None: real LLMClient -> StrictLLMClient; offline mocks are kept as they are.
_strict_client_factory: Optional[Callable[[], object]] = None


def set_strict_client_factory(factory: Optional[Callable[[], object]]) -> None:
    global _strict_client_factory
    _strict_client_factory = factory


class ExperimentPlayer(Player):
    """LLM player used only by the scripted-offer experiment (GameScriptedOffer).

    Keeps the original Player untouched. Differences from Player, all confined to this class:
      * private_messages: text appended verbatim to play and challenge prompts (the scripted
        offer and the agent's own reply); empty unless the experiment sets it;
      * last_play_fallback / last_challenge_fallback and an "is_fallback" key on fallback
        results, so fallback decisions can be identified and excluded from the analysis;
      * duplicate-card validation by card counts. The original per-card membership check let a
        play such as ["A", "A"] from a hand with one A pass, and the removal loop then deleted
        cards before failing; here such a play is rejected and the model is asked again;
      * the real client is a StrictLLMClient, and ApiRequestError is re-raised by every method
        that calls the model, so an API failure ends the game instead of becoming a fallback.
    The three methods below are the original Player methods with only these changes applied.
    """

    def __init__(self, name: str, model_name: str):
        super().__init__(name, model_name)
        if _strict_client_factory is not None:
            self.llm_client = _strict_client_factory()
        elif isinstance(self.llm_client, LLMClient):
            self.llm_client = StrictLLMClient()
        if hasattr(self.llm_client, "owner"):
            self.llm_client.owner = name
        self.private_messages = ""
        self.last_play_fallback = False
        self.last_challenge_fallback = False

    @property
    def api_retries(self) -> int:
        return getattr(self.llm_client, "api_retries", 0)

    def choose_cards_to_play(self,
                        round_base_info: str,
                        round_action_info: str,
                        play_decision_info: str) -> Dict:
        """
        Play cards based on the given information
        
        Args:
            round_base_info: basic information of the round
            round_action_info: round action information
            play_decision_info: play decision information
            
        Returns:
            tuple: (result, reasoning_content)
            - result dictionary includes played_cards, behavior and play_reason
            - The reasoning content is the original reasoning process of the LLM
        """
        # Read rules and templates
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(PLAY_CARD_PROMPT_TEMPLATE_PATH)
        
        # Current cards
        current_cards = ", ".join(self.hand)
        
        # Add series memory context if available and non-empty
        memory_section = ""
        if self.series_memory and self.series_memory.strip():
            memory_section = f"\n\n=== CROSS-GAME MEMORY ===\n{self.series_memory}\n=== END MEMORY ===\n"
        
        # Add secret alliance memory context - CRITICAL FIX
        if self.secret_channel_memory.get('enabled') and self.secret_channel_memory.get('partner_name'):
            partner = self.secret_channel_memory['partner_name']
            alliance_memory = self.get_secret_alliance_memory(partner, game_number=1, round_number=1)
            memory_section += alliance_memory

        if self.private_messages:
            memory_section += self.private_messages
        
        # Fill in the template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            play_decision_info=play_decision_info,
            current_cards=current_cards
        ) + memory_section
        
        # Try to get a valid JSON response, up to five times more. On each retry we append a
        # specific description of what was wrong with the previous attempt so the model can
        # correct it, instead of re-sending an identical prompt (which can loop indefinitely).
        error_feedback = ""
        for attempt in range(5):
            messages = [
                {"role": "user", "content": prompt + error_feedback}
            ]

            try:
                content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)

                # Parse the JSON response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)

                    # Verify that the JSON format meets the requirements
                    if all(key in result for key in ["played_cards", "behavior", "play_reason"]):
                        # Ensure that played_cards is a list
                        if not isinstance(result["played_cards"], list):
                            result["played_cards"] = [result["played_cards"]]

                        # Ensure that the played cards are valid
                        # Multiset check: each card may be played at most as many times as it is
                        # held. A per-card membership test let duplicates such as ["A", "A"] from a
                        # hand with one A pass, and the removal loop then deleted cards before failing.
                        played_counts = Counter(result["played_cards"])
                        hand_counts = Counter(self.hand)
                        valid_cards = all(hand_counts[c] >= n for c, n in played_counts.items())
                        valid_count = 1 <= len(result["played_cards"]) <= 3

                        if valid_cards and valid_count:
                            # remove the played cards from the hand
                            for card in result["played_cards"]:
                                self.hand.remove(card)
                            self.last_play_fallback = False
                            return result, reasoning_content
                        else:
                            # Tell the model exactly why its move was rejected so the next attempt differs.
                            reasons = []
                            if not valid_cards:
                                bad = [c for c, n in played_counts.items() if hand_counts[c] < n]
                                reasons.append(f"you tried to play card(s) not in your hand: {bad}. "
                                               f"Your hand is: {', '.join(self.hand)}")
                            if not valid_count:
                                reasons.append(f"you must play between 1 and 3 cards, but you played "
                                               f"{len(result['played_cards'])}")
                            error_feedback = ("\n\nYour previous response was INVALID because "
                                              + "; ".join(reasons)
                                              + ". Please respond again with a corrected, valid move.")
                            print(f"Try {attempt+1} invalid: {'; '.join(reasons)}")
                    else:
                        error_feedback = ("\n\nYour previous response was INVALID: the JSON must contain "
                                          "the keys 'played_cards', 'behavior', and 'play_reason'. "
                                          "Please respond again with valid JSON.")
                        print(f"Try {attempt+1} invalid: missing required JSON keys")
                else:
                    error_feedback = ("\n\nYour previous response was INVALID: no JSON object was found. "
                                      "Please respond with a single valid JSON object.")
                    print(f"Try {attempt+1} invalid: no JSON object found")

            except ApiRequestError:
                raise
            except Exception as e:
                # Record the error and feed it back so the retry prompt differs.
                error_feedback = ("\n\nYour previous response could not be parsed as valid JSON "
                                  f"({str(e)}). Please respond with a single valid JSON object.")
                print(f"Try {attempt+1} failed: {str(e)}")
        # Graceful fallback after repeated unparseable/invalid LLM responses: play a single card rather
        # than hard-raising, which would abort the whole game before a winner (small models occasionally
        # emit empty/invalid JSON several turns in a row). Mirrors the Cleanup agent's default-action
        # behavior so one bad seat-response doesn't crash a long game.
        if self.hand:
            print(f"Player {self.name}'s choose_cards_to_play failed after 5 attempts; "
                  f"defaulting to playing one card.")
            default_card = self.hand.pop(0)
            result = {
                "played_cards": [default_card],
                "behavior": "",
                "play_reason": "fallback: no valid LLM response after repeated attempts",
                "is_fallback": True,
            }
            self.last_play_fallback = True
            return result, ""
        raise RuntimeError(f"Player {self.name}'s choose_cards_to_play failed after many attempts and has no cards to fall back on.")

    def decide_challenge(self,
                        round_base_info: str,
                        round_action_info: str,
                        challenge_decision_info: str,
                        challenging_player_performance: str,
                        extra_hint: str) -> bool:
        """
        Decide whether to challenge the previous player's card
        
        Args:

            round_base_info: basic information of the round
            round_action_info: round action information
            challenge_decision_info: challenge decision information
            challenging_player_performance: performance of the challenging player
            extra_hint: additional hints
            
        Returns:
            tuple: (result, reasoning_content)
            - result: A dictionary contains was_challenged and challenge_reason
            - reasoning_content: LLM's original reasoning process
        """
        # read rules and templates
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(CHALLENGE_PROMPT_TEMPLATE_PATH)
        self_hand = f"Your current hand is: {', '.join(self.hand)}"
        
        # Add series memory context if available and non-empty
        memory_section = ""
        if self.series_memory and self.series_memory.strip():
            memory_section = f"\n\n=== CROSS-GAME MEMORY ===\n{self.series_memory}\n=== END MEMORY ===\n"
        
        # Add secret alliance memory context - CRITICAL FIX
        if self.secret_channel_memory.get('enabled') and self.secret_channel_memory.get('partner_name'):
            partner = self.secret_channel_memory['partner_name']
            alliance_memory = self.get_secret_alliance_memory(partner, game_number=1, round_number=1)
            memory_section += alliance_memory

        if self.private_messages:
            memory_section += self.private_messages
        
        # fill in the template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            self_hand=self_hand,
            challenge_decision_info=challenge_decision_info,
            challenging_player_performance=challenging_player_performance,
            extra_hint=extra_hint
        ) + memory_section
        
        # try to get a valid JSON response, up to five times more
        for attempt in range(5):
            # send the same prompt each time
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                
                # parse the JSON response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    
                    # verify that the JSON format meets the requirements
                    if all(key in result for key in ["was_challenged", "challenge_reason"]):
                        # ensure that was_challenged is a boolean
                        if isinstance(result["was_challenged"], bool):
                            self.last_challenge_fallback = False
                            return result, reasoning_content

            except ApiRequestError:
                raise
            except Exception as e:
                # only record errors without modifying retry requests
                print(f"Attemp {attempt+1} failed: {str(e)}")
        # Graceful fallback after repeated unparseable/invalid LLM responses: default to NOT challenging
        # rather than hard-raising, which would abort the whole game (small models occasionally emit
        # empty/invalid JSON several turns in a row). Mirrors choose_cards_to_play's default-action
        # behavior so one bad seat-response doesn't crash a long game.
        print(f"Player {self.name}'s decide_challenge failed after 5 attempts; "
              f"defaulting to not challenging.")
        result = {
            "was_challenged": False,
            "challenge_reason": "fallback: no valid LLM response after repeated attempts",
            "is_fallback": True,
        }
        self.last_challenge_fallback = True
        return result, ""

    def reflect(self, alive_players: List[str], round_base_info: str, round_action_info: str, round_result: str) -> None:
        """
        Reflect on the game and update the impression of other players

        Args:
            alive_players: list of alive players
            round_base_info: basic information of the round
            round_action_info: round action information
            round_result: round result
        """
        # read templates
        template = self._read_file(REFLECT_PROMPT_TEMPLATE_PATH)

        # read rules
        rules = self._read_file(RULE_BASE_PATH)

        # reflect on each player
        for player_name in alive_players:
            # skip self
            if player_name == self.name:
                continue

            # get the previous opinion
            previous_opinion = self.opinions.get(player_name, "Doesn't know well about this player yet")

            # Add series memory context if available
            memory_section = ""
            if self.series_memory:
                memory_section = f"\n\n=== CROSS-GAME MEMORY ===\n{self.series_memory}\n=== END MEMORY ===\n"

            # fill in the template
            prompt = template.format(
                rules=rules,
                self_name=self.name,
                round_base_info=round_base_info,
                round_action_info=round_action_info,
                round_result=round_result,
                player=player_name,
                previous_opinion=previous_opinion
            ) + memory_section

            # try to get a valid JSON response, up to five times more
            messages = [
                {"role": "user", "content": prompt}
            ]

            try:
                content, _ = self.llm_client.chat(messages, model=self.model_name)

                # update the opinion
                self.opinions[player_name] = content.strip()
                print(f"{self.name} updated the impression to player {player_name}")

            except ApiRequestError:
                raise
            except Exception as e:
                print(f"Error while reflecting {player_name} : {str(e)}")
