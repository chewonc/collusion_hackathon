"""Offline checks for the scripted-offer experiment (no network; uses MockLLMClient).

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_scripted_offer.py
"""
import contextlib
import io
import os
import sys
import tempfile
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.agents import player as player_module
from src.test.mock_llm import MockLLMClient

player_module.LLMClient = MockLLMClient

from src.game.game_scripted_offer import (AGENT_CONFIGS, CONDITIONS, GameScriptedOffer,
                                          parse_offer_response, seat_order)

BLOCK = "=== PRIVATE MESSAGES ==="
REMINDER = "DO NOT CHALLENGE"


def run_game(cond, g, out_dir):
    MockLLMClient.calls = []
    with contextlib.redirect_stdout(io.StringIO()):
        game = GameScriptedOffer(cond, g, seed=g, out_dir=out_dir)
        record = game.start_game()
    return record, list(MockLLMClient.calls)


def round1_state(record):
    r1 = record["rounds"][0]
    return (r1["target_card"], r1["starting_player"],
            [(s["player_name"], s["bullet_position"], sorted(s["initial_hand"])) for s in r1["player_initial_states"]])


def strip_volatile(record):
    record = dict(record)
    record.pop("game_id", None)
    return record


def test_seat_rotation():
    first = Counter(seat_order(g)[0]["name"] for g in range(12))
    assert first == Counter({c["name"]: 4 for c in AGENT_CONFIGS}), first
    for g in range(12):
        assert seat_order(g) == seat_order(g)
        assert sorted(c["name"] for c in seat_order(g)) == sorted(c["name"] for c in AGENT_CONFIGS)
    print("ok seat rotation: each agent sits after the proposer 4 times in 12 games")


def test_parser():
    cases = {
        '{"decision": "ACCEPT", "reply": "Deal."}': "ACCEPT",
        'Sure! {"decision": "decline", "reply": "No thanks."}': "DECLINE",
        '{"decision": "MAYBE", "reply": "Hmm."}': "UNREADABLE",
        '{"decision": "ACCEPT"}': "UNREADABLE",
        'ACCEPT - sounds good': "UNREADABLE",
        '': "UNREADABLE",
        '{"decision": "ACCEPT", "reply": "x"': "UNREADABLE",
    }
    for text, expected in cases.items():
        got = parse_offer_response(text)["decision"]
        assert got == expected, (text, got, expected)
    print(f"ok parser: {len(cases)} cases")


def test_reproducibility_and_injection(out_dir):
    for g in (0, 1, 2):
        states = {}
        for cond in CONDITIONS:
            record, calls = run_game(cond, g, out_dir)
            states[cond] = round1_state(record)
            check_injection(cond, record, calls)
            check_fallback_flags(record)
        assert len(set(map(repr, states.values()))) == 1, f"round-1 state differs across conditions for g={g}"
    print("ok same seed -> identical round-1 deal, bullets, start seat and target across all 5 conditions (g=0,1,2)")

    a, _ = run_game("C1", 3, out_dir)
    b, _ = run_game("C1", 3, out_dir)
    assert strip_volatile(a) == strip_volatile(b), "same condition + seed + deterministic model must replay exactly"
    print("ok same condition + seed + deterministic mock -> identical full record")


def check_injection(cond, record, calls):
    so = record["scripted_offer"]
    offers = so["offers"]
    names = [c["name"] for c in AGENT_CONFIGS]
    offer_idx = {}
    for i, (_, prompt) in enumerate(calls):
        if '"decision"' in prompt:
            for n in names:
                if f"You are {n}." in prompt:
                    offer_idx[n] = i
    if cond == "C4":
        assert offers == [] and offer_idx == {}, "C4 must send no messages"
        assert not any(BLOCK in p for _, p in calls), "C4 prompts must never contain the private block"
        assert not any(REMINDER in p for _, p in calls)
        return
    if so["offer_round_actual"] is None:
        return
    expected = {so["c0_ally"]} if cond == "C0" else {o["agent"] for o in offers}
    assert {o["agent"] for o in offers} == set(offer_idx) == expected, (cond, offers, offer_idx)
    for i, (_, prompt) in enumerate(calls):
        is_decision = '"played_cards"' in prompt or '"was_challenged"' in prompt
        if not is_decision:
            continue
        who = next((n for n in names if f"you are {n}\n" in prompt), None)
        assert who is not None
        has_block = BLOCK in prompt
        if who in offer_idx and i > offer_idx[who]:
            assert has_block, (cond, who, "missing private block after offer")
        else:
            assert not has_block, (cond, who, "private block present before offer / for non-recipient")
        if REMINDER in prompt:
            assert cond == "C0" and who == so["c0_ally"] and '"was_challenged"' in prompt
    if cond == "C0":
        ally_rows = [d for d in so["decisions"] if d["challenger"] == so["c0_ally"]
                     and d["is_proposer_play"] and d["post_offer"]]
        ally_reminders = sum(1 for _, p in calls if REMINDER in p)
        assert ally_reminders >= len(ally_rows), "C0 ally must get the alliance reminder on every proposer play"


def check_fallback_flags(record):
    so = record["scripted_offer"]
    for d in so["decisions"]:
        if d["is_system"] or d["challenger_is_scripted"]:
            continue
        is_fb_reason = (d["challenge_reason"] or "").startswith("fallback:")
        assert d["is_fallback"] == is_fb_reason, d
    reasons = [a["play_reason"] for r in record["rounds"] for a in r["play_history"]
               if a["play_reason"] != "You are the last player, cards will be automatically played."]
    play_fb = [p["is_fallback"] for p in so["plays"]]
    assert play_fb == [r.startswith("fallback:") for r in reasons], "play fallback flags out of sync"


def test_proposer_behaviour(out_dir):
    record, _ = run_game("C4", 5, out_dir)
    so = record["scripted_offer"]
    prop = so["proposer"]
    for d in so["decisions"]:
        if d["challenger"] == prop:
            assert not d["was_challenged"], "proposer must never challenge"
    pplays = [p for p in so["plays"] if p["player"] == prop]
    assert all(len(p["played_cards"]) == 1 for p in pplays)
    print(f"ok proposer: never challenges, always plays one card ({len(pplays)} plays, "
          f"{sum(p['is_bluff'] for p in pplays)} bluffs)")


def test_no_double_reset(out_dir):
    """Our game must award the last-survivor bonus once and start one round per round-ending event.

    The upstream GameSecretChannel.play_round re-runs check_victory/reset_round after
    perform_penalty, which awards the bonus twice and deals an extra round; GameScriptedOffer
    overrides play_round to avoid that.
    """
    for cond in ("C0", "C4"):
        for g in range(4):
            with contextlib.redirect_stdout(io.StringIO()):
                game = GameScriptedOffer(cond, g, seed=g, out_dir=out_dir)
                counts = {"victory": 0, "penalty": 0, "system": 0, "rounds": 0}
                cv, pp, hs, sr = (game.check_victory, game.perform_penalty, game.handle_system_challenge,
                                  game.start_round_record)

                def check_victory():
                    r = cv()
                    counts["victory"] += bool(r)
                    return r

                def perform_penalty(*a, **k):
                    counts["penalty"] += 1
                    return pp(*a, **k)

                def handle_system_challenge(*a, **k):
                    counts["system"] += 1
                    return hs(*a, **k)

                def start_round_record():
                    counts["rounds"] += 1
                    return sr()

                game.check_victory, game.perform_penalty = check_victory, perform_penalty
                game.handle_system_challenge, game.start_round_record = handle_system_challenge, start_round_record
                game.start_game()
            assert counts["victory"] == 1, (cond, g, counts)
            # Each round after the first starts from one penalty or one system challenge, never two.
            assert counts["rounds"] <= 1 + counts["penalty"] + counts["system"], (cond, g, counts)
    print("ok no double reset: last-survivor bonus awarded once; one new round per round-ending event")


def _play(cls, hand, replies):
    """Run one choose_cards_to_play call of `cls` with scripted model replies; return (result, hand, error)."""
    it = iter(replies)

    class Scripted:
        def chat(self, messages, model=None):
            return next(it, '{"played_cards": [], "behavior": "b", "play_reason": "r"}'), ""

    p = cls("Luke", "llama-3.2-3b")
    p.llm_client = Scripted()
    p.hand = list(hand)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            result, _ = p.choose_cards_to_play("info", "actions", "decision")
            return result, p.hand, None
        except Exception as e:
            return None, p.hand, e


def _reply(cards):
    return '{"played_cards": %s, "behavior": "b", "play_reason": "r"}' % str(cards).replace("'", '"')


def test_duplicate_cards_rejected_without_losing_cards():
    """Duplicate-card validation by counts, replaying the two incidents of the crashed C4 pilot game."""
    from src.agents.experiment_player import ExperimentPlayer
    Original = player_module.Player

    # Incident 1 (Quinn): hand A,A,K,Joker,Q; model asked for ["K","K"], then played ["A"].
    res, hand, err = _play(Original, ["A", "A", "K", "Joker", "Q"], [_reply(["K", "K"]), _reply(["A"])])
    assert err is None and sorted(hand) == ["A", "Joker", "Q"], ("original must silently lose the K", hand)
    res, hand, err = _play(ExperimentPlayer, ["A", "A", "K", "Joker", "Q"], [_reply(["K", "K"]), _reply(["A"])])
    assert err is None and res["played_cards"] == ["A"] and sorted(hand) == ["A", "Joker", "K", "Q"], hand

    # Incident 2 (Luke): hand K,A; model asked for ["K","A","K"], then ["K","A","A"], then nothing valid.
    seq = [_reply(["K", "A", "K"]), _reply(["K", "A", "A"])] + [_reply(["K", "K"])] * 3
    res, hand, err = _play(Original, ["K", "A"], seq)
    assert isinstance(err, RuntimeError) and hand == [], ("original must empty the hand and crash", hand, err)
    res, hand, err = _play(ExperimentPlayer, ["K", "A"], seq)
    assert err is None and res["is_fallback"] and len(res["played_cards"]) == 1 and len(hand) == 1, (res, hand)

    # Valid duplicates are accepted and removed exactly.
    res, hand, err = _play(ExperimentPlayer, ["A", "A", "K"], [_reply(["A", "A"])])
    assert err is None and res["played_cards"] == ["A", "A"] and hand == ["K"]
    res, hand, err = _play(ExperimentPlayer, ["Joker", "A", "A", "Q"], [_reply(["A", "Joker", "A"])])
    assert err is None and hand == ["Q"]
    # Cards not held and too many cards are still rejected without touching the hand.
    res, hand, err = _play(ExperimentPlayer, ["K", "A"], [_reply(["Q"]), _reply(["K"])])
    assert res["played_cards"] == ["K"] and hand == ["A"]
    res, hand, err = _play(ExperimentPlayer, ["A", "A", "A", "K"], [_reply(["A", "A", "A", "K"]), _reply(["K"])])
    assert res["played_cards"] == ["K"] and hand == ["A", "A", "A"]

    # The experiment uses ExperimentPlayer for every LLM seat in every condition.
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        for c in CONDITIONS:
            game = GameScriptedOffer(c, 0, seed=0, out_dir=tmp)
            llm = [p for p in game.players if p is not game.proposer]
            assert len(llm) == 3 and all(type(p) is ExperimentPlayer for p in llm), c
    print("ok duplicate cards: both C4-pilot incidents reproduced on the original Player (silent loss, crash) "
          "and fixed in ExperimentPlayer; valid duplicates accepted; all conditions use ExperimentPlayer")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_duplicate_cards_rejected_without_losing_cards()
        test_no_double_reset(tmp)
        test_seat_rotation()
        test_parser()
        test_reproducibility_and_injection(tmp)
        print("ok injection: offered agents get the verbatim private block on every later play/challenge "
              "prompt, never before; C0 only the ally (+ reminder); C4 nobody")
        print("ok fallback flags match the recorded fallback reasons")
        test_proposer_behaviour(tmp)
    print("ALL OFFLINE CHECKS PASSED")
