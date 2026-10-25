"""Offline checks for the scripted-offer experiment (no network; uses MockLLMClient).

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_scripted_offer.py
"""
import contextlib
import io
import os
import re
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
    print(f"ok same seed -> identical round-1 deal, bullets, start seat and target across all {len(CONDITIONS)} "
          "conditions (g=0,1,2)")

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


def test_placebo(out_dir):
    """C5: same round-1 state, same recipients, timing, response prompt and injected block as C1-C3;
    only the message text differs, and it proposes nothing."""
    from src.utils.prompt_loader import PromptLoader
    placebo = PromptLoader.load_prompt("prompts/scripted_offer/offer_placebo.txt")
    benign = PromptLoader.load_prompt("prompts/scripted_offer/offer_benign.txt")
    hint = PromptLoader.load_prompt("prompts/scripted_offer/offer_hint.txt")
    pact_words = ("challenge", "deal", "trust each other", "let each other", "your plays", "my plays",
                  "alliance", "work together", "friendly")
    for name, text in (("C5", placebo), ("C3", benign), ("C2", hint)):
        body = text.split("\n", 1)[1].lower()   # the message, without the shared header line
        for w in pact_words:
            assert not re.search(r"\b" + re.escape(w) + r"\b", body), f"{name} must not mention a pact: {w!r}"
    for name, text in (("C5", placebo), ("C2", hint)):
        assert "agree" not in text.split("\n", 1)[1].lower(), f"{name} must not ask for agreement"
    body5 = placebo.split("\n", 1)[1].lower()
    for w in ("?", "let's", "shall we", "how about", "good luck", "together", "help", "enjoy"):
        assert w not in body5, f"C5 must contain no request or encouragement: {w!r}"
    body3 = benign.split("\n", 1)[1].lower()
    # v7 C3: asks for agreement ("can we agree") that the game stays enjoyable; nothing about
    # challenges, plays, alliances or any player's treatment.
    assert "?" in body3 and "agree" in body3 and "enjoyable" in body3, "C3 must be an explicit benign request"
    assert "?" not in hint.split("\n", 1)[1], "C2 must stay an indirect statement, not a request for agreement"
    for g in range(6):
        recs = {c: run_game(c, g, out_dir)[0] for c in ("C1", "C2", "C3", "C5")}
        assert len({repr(round1_state(r)) for r in recs.values()}) == 1, f"g{g}: C5 round-1 state differs"
        so = {c: r["scripted_offer"] for c, r in recs.items()}
        rnd = so["C5"]["offer_round_actual"]
        if rnd is None:
            continue
        alive = set(next(x for x in recs["C5"]["rounds"] if x["round_id"] == rnd)["round_players"]) - {"Riley"}
        recipients = {c: {o["agent"] for o in so[c]["offers"]} for c in so}
        assert recipients["C5"] == alive, f"g{g}: C5 must message every alive agent"
        assert all(not o["agent"] == "Riley" for c in so for o in so[c]["offers"])
        assert all(recipients[c] == recipients["C5"] for c in ("C1", "C2", "C3")), f"g{g}: {recipients}"
        assert {so[c]["offer_round_actual"] for c in so} == {rnd}, f"g{g}: offer round differs"
        for o5 in so["C5"]["offers"]:
            o1 = next(o for o in so["C1"]["offers"] if o["agent"] == o5["agent"])
            assert o5["offer_text"] == placebo.replace("{proposer_name}", "Riley").replace(
                "{agent_name}", o5["agent"]).strip()
            # Same response prompt and same injected block: identical once the message text is swapped.
            assert o5["prompt"] == o1["prompt"].replace(o1["offer_text"], o5["offer_text"])
            def shape(o):
                # Same block with the message and the agent's own reply line masked (the reply may be
                # the raw text of an UNREADABLE answer, so the whole "you to Riley" line is masked).
                return re.sub(r'(?m)^Round \d+, you to Riley: .*$', "<R>",
                              o["injected_block"].replace(o["offer_text"], "<MSG>"))
            assert shape(o5) == shape(o1), "injected block must have the same form as in C1"
    print("ok C5 placebo: same round-1 state, recipients, offer round, response prompt and injected block as C1-C3; "
          "only the (pact-free) message differs")


def test_broker_c1b(out_dir):
    """C1b (v7): only the alive seat after the proposer gets the brokered message; first breach and
    after_first_breach are recorded as specified; adding C1b leaves every other condition's record
    byte-identical (each game reseeds from its own seed)."""
    from src.utils.prompt_loader import PromptLoader
    broker = PromptLoader.load_prompt("prompts/scripted_offer/offer_broker.txt")
    body = broker.split("\n", 1)[1]
    assert broker.startswith("Private message from {proposer_name} (only you can see it):\n")
    assert "other two players" in body and "don't challenge my plays" in body and "Deal?" in body
    assert "{agent_name}" in body and "{" not in body.replace("{agent_name}", ""), "no extra placeholders"
    n_breach = n_games_with_x = 0
    for g in range(6):
        rec, calls = run_game("C1b", g, out_dir)
        so = rec["scripted_offer"]
        assert so["condition_name"] == "false_brokered_offer"
        if so["offer_round_actual"] is None:
            continue
        rnd = next(x for x in rec["rounds"] if x["round_id"] == so["offer_round_actual"])
        alive = [p for p in so["seat_order"] if p in rnd["round_players"]]
        if so["target"] is None:
            # Proposer dead at the offer: the record says so (no breach can be recorded).
            assert so["first_breach"] is None
        else:
            assert so["target"] == so["c1b_recipient"] == alive[1 if alive[0] == "Riley" else 0], (g, alive, so["target"])
            n_games_with_x += 1
        recipients = {o["agent"] for o in so["offers"]}
        assert recipients == ({so["c1b_recipient"]} if so["c1b_recipient"] else set()), (g, recipients)
        # Nobody else ever sees the private block.
        for i, (_, prompt) in enumerate(calls):
            if BLOCK in prompt:
                assert f"you are {so['c1b_recipient']}\n" in prompt or f"You are {so['c1b_recipient']}." in prompt, g
        fb = so["first_breach"]
        flags = [d["after_first_breach"] for d in so["decisions"]]
        if fb is None:
            assert not any(flags)
            # No post-offer LLM challenge of the recipient's play may exist when target is set.
            if so["target"] is not None:
                assert not any(d["post_offer"] and d["current_player"] == so["c1b_recipient"] and d["was_challenged"]
                               and not d["is_system"] and not d["challenger_is_scripted"] for d in so["decisions"])
        else:
            n_breach += 1
            i = fb["decision_index"]
            d = so["decisions"][i]
            assert d["current_player"] == so["c1b_recipient"] and d["was_challenged"] and d["post_offer"]
            assert not d["is_system"] and not d["challenger_is_scripted"] and d["challenger"] == fb["challenger"]
            assert fb["challenge_succeeded"] == d["play_was_bluff"] and fb["round"] == d["round"]
            assert flags == [False] * i + [True] * (len(flags) - i), "after_first_breach must flip once"
            earlier = [e for e in so["decisions"][:i] if e["post_offer"] and e["current_player"] == so["c1b_recipient"]
                       and e["was_challenged"] and not e["is_system"] and not e["challenger_is_scripted"]]
            assert not earlier, "an earlier breach was missed"
    assert n_games_with_x >= 3, "mock games should give some C1b games with a live proposer at the offer"
    # Other conditions are unaffected by C1b's existence: their records never mention it and are
    # identical to a run of the condition list without C1b (each game reseeds; no shared state).
    for g in (0, 1):
        for c in ("C1", "C4", "C5"):
            rec, _ = run_game(c, g, out_dir)
            assert rec["scripted_offer"]["c1b_recipient"] is None and rec["scripted_offer"]["first_breach"] is None
            assert not any(d["after_first_breach"] for d in rec["scripted_offer"]["decisions"])
            again, _ = run_game(c, g, out_dir)
            assert strip_volatile(rec) == strip_volatile(again)
    print(f"ok C1b: recipient = alive seat after the proposer only; first breach recorded in {n_breach} of 6 mock "
          "games with correct index/round/challenger; after_first_breach flips once; other conditions unchanged")


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
        test_placebo(tmp)
        test_broker_c1b(tmp)
    print("ALL OFFLINE CHECKS PASSED")
