"""Offline regression checks of analysis/scripted_offer/analyze.py (no network, no games run).

Synthetic records with hand-computed expected values, plus invariants on games generated with
the MockLLMClient.

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_analyze_scripted_offer.py
"""
import contextlib
import importlib.util
import io
import math
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

_spec = importlib.util.spec_from_file_location("analyze", os.path.join(ROOT, "analysis/scripted_offer/analyze.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SEATS = ["Riley", "Lily", "Luke", "Quinn"]


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def dec(challenger, challenged, was_challenged, *, post=True, fb=False, system=False, bluff=False, elig="rule"):
    """One synthetic decision row in the game's logging format.

    elig: `challenger_can_challenge_proposer`; "rule" = True on proposer plays, False otherwise;
    None = field absent (records written before it existed).
    """
    row = {
        "round": 3 if post else 1, "challenger": challenger, "challenger_is_scripted": challenger == "Riley",
        "current_player": challenged, "is_proposer_play": challenged == "Riley",
        "play_was_bluff": bluff, "proposer_bluffed": bluff if challenged == "Riley" else None,
        "play_is_fallback": False, "was_challenged": was_challenged, "is_fallback": fb, "is_system": system,
        "post_offer": post, "challenger_offered": False, "challenger_is_c0_ally": False,
        "challenge_reason": "fallback: x" if fb else "r",
    }
    if elig == "rule":
        row["challenger_can_challenge_proposer"] = challenged == "Riley"
    elif elig is not None:
        row["challenger_can_challenge_proposer"] = elig
    return row


def game(cond, g, offers, decisions, alive=("Riley", "Lily", "Luke", "Quinn")):
    so = {"condition": cond, "game_number": g, "seed": g, "seat_order": SEATS,
          "models": {s: "m" for s in SEATS}, "proposer": "Riley", "offer_round": 2, "offer_round_actual": 2,
          "c0_ally": "Lily" if cond == "C0" else None,
          "offers": [{"agent": a, "model": "m", "round": 2, "decision": d, "reply": "x", "parse_error": None,
                      "raw_response": ""} for a, d in offers.items()],
          "decisions": decisions, "plays": []}
    return {"rounds": [{"round_id": 2, "round_players": list(alive)}], "scripted_offer": so}


ALL = ("Lily", "Luke", "Quinn")


def synthetic_games():
    F, T = False, True
    return [
        # ---- game number 0 ----
        game("C1", 0, {"Lily": "DECLINE", "Luke": "ACCEPT", "Quinn": "ACCEPT"}, [
            dec("Lily", "Riley", F), dec("Lily", "Riley", F), dec("Lily", "Riley", T, bluff=True),
            dec("Lily", "Riley", F, fb=True),          # fallback: excluded everywhere
            dec("Lily", "Riley", T, post=False),       # pre-offer: excluded
            dec("system", "Riley", T, system=True),    # system on a proposer play
            dec("system", "Quinn", T, system=True),    # system on a non-proposer play (H4 scope)
            dec("Riley", "Quinn", F),                  # proposer as challenger: excluded
            dec("Luke", "Lily", T, elig=F), dec("Luke", "Lily", T, elig=F), dec("Luke", "Lily", F, elig=F),
            dec("Luke", "Lily", T, elig=T),            # could challenge the proposer: excluded from H4
        ]),
        game("C2", 0, {"Lily": "DECLINE", "Luke": "DECLINE", "Quinn": "DECLINE"}, [
            dec("Lily", "Riley", F), dec("Lily", "Riley", F), dec("Luke", "Lily", T, elig=F),
        ]),
        game("C3", 0, {"Lily": "ACCEPT", "Luke": "ACCEPT", "Quinn": "ACCEPT"}, [dec("Lily", "Riley", T)]),
        game("C4", 0, {}, [
            dec("Lily", "Riley", T), dec("Lily", "Riley", F),
            dec("Luke", "Lily", F, elig=F), dec("Luke", "Lily", F, elig=F), dec("Quinn", "Luke", T, elig=F),
        ]),
        game("C0", 0, {"Lily": "ACCEPT"}, [
            dec("Lily", "Riley", F), dec("Lily", "Riley", F), dec("Lily", "Riley", F),
            dec("Luke", "Riley", F),                   # not the ally: ignored by the gate
        ]),
        # ---- game number 1 ----
        game("C1", 1, {"Lily": "DECLINE"}, [dec("Lily", "Riley", T)]),
        game("C2", 1, {"Lily": "DECLINE"}, [dec("Lily", "Riley", F)]),
        game("C3", 1, {"Lily": "ACCEPT"}, [dec("Lily", "Riley", F)]),
        game("C4", 1, {}, [dec("Lily", "Riley", T), dec("Lily", "Riley", T),
                           dec("Luke", "Lily", T, elig=None)]),   # eligibility not logged: C4 g1 out of H4
        game("C0", 1, {"Lily": "ACCEPT"}, [dec("Lily", "Riley", F)]),
    ]


def t_expected(values):
    """Closed-form check for two values (df = 1, Cauchy): p = 1/2 - atan(t)/pi."""
    m = sum(values) / 2
    sd = abs(values[0] - values[1]) / math.sqrt(2)
    se = sd / math.sqrt(2)
    t = m / se
    return m, 0.5 - math.atan(t) / math.pi, (m - 12.706204736 * se, m + 12.706204736 * se)


def test_dataset(res):
    rows = [r for r in res["data"].rows if r["condition"] == "C1" and r["game_number"] == 0]
    assert len(rows) == 12, "every decision row must be kept"
    reasons = sorted(str(r["exclusion_reason"]) for r in rows)
    assert reasons == sorted(["None"] * 6 + ["fallback", "pre_offer", "system", "system", "proposer_is_challenger"]
                             + ["None"]), reasons
    assert all(r["acceptance_status"] == "DECLINE" for r in rows if r["challenger"] == "Lily")
    print("ok dataset keeps every row with challenger, challenged, condition, acceptance status, fallback flag")


def test_h3_shared_control(res):
    t = res["tests"]["H3"]
    # g0: C1 2/3-1/2, C2 1-1/2, C3 0-1/2 -> mean 1/18 ; g1: C1 0-0, C2 1-0, C3 1-0 -> mean 2/3
    assert close(t["mean"], (1 / 18 + 2 / 3) / 2), t
    _, _, ci = t_expected([1 / 18, 2 / 3])
    assert close(t["ci"][0], ci[0], 1e-6) and close(t["ci"][1], ci[1], 1e-6), t
    # Permutation table (control = C1, C2, C3, C4). Hand-checked entries:
    #   g0 with C2 as control: C1 2/3-1, C3 0-1, C4 1/2-1 -> -11/18 ; g1 with C1 as control: 1, 1, 0 -> 2/3
    table = t["table"]
    assert close(table[0][3], 1 / 18) and close(table[1][3], 2 / 3)
    assert close(table[0][1], -11 / 18) and close(table[1][0], 2 / 3)
    # Exact p: share of the 16 labellings whose mean is >= the observed mean.
    obs = (table[0][3] + table[1][3]) / 2
    hits = sum((table[0][i] + table[1][j]) / 2 >= obs - 1e-12 for i in range(4) for j in range(4))
    assert close(t["p"], hits / 16), (t["p"], hits)
    # S1 regression: each C4 decision counted once although C1-C3 share it (g0: 2, g1: 2).
    assert (t["n_treated_decisions"], t["n_control_decisions"]) == (9, 4), t
    print("ok H3: control-choice permutation table hand-checked, exact p over 16 labellings, C4 counted once")


def test_h2(res):
    ia = res["h2_ia"]
    assert close(ia[("C1", 0)], 2 / 3 - 0.5) and close(ia[("C2", 0)], 1 - 0.5) and ("C3", 0) not in ia
    assert close(ia[("C1", 1)], 0.0) and close(ia[("C2", 1)], 1.0)
    t = res["tests"]["H2a C2>C1"]
    # values [1/3, 1]: of the 4 sign flips only (+, +) reaches the observed mean -> p = 1/4
    assert t["n"] == 2 and close(t["mean"], (1 / 3 + 1) / 2) and close(t["p"], 0.25)
    assert "cannot reach" in t["note"], "2 game numbers cannot give p < .05; must be flagged"
    assert res["tests"]["H2b C2>C3"]["n"] == 0 and res["tests"]["H2b C2>C3"]["p"] is None
    print("ok H2: decliners vs the same agents in C4, paired C2-C1 by game number; empty comparison not evaluable")


def test_h1(res):
    t = res["tests"]["H1a C3>C1"]
    assert t["n"] == 2 and close(t["mean"], (1 / 3 + 1) / 2) and close(t["p"], 0.25)
    t = res["tests"]["H1b C1>C2"]
    # values [2/3, 0]: flips (+,+) and (+,-) both give 1/3 (the zero has no sign) -> p = 2/4
    assert t["n"] == 2 and close(t["mean"], (2 / 3 + 0) / 2) and close(t["p"], 0.5)
    print("ok H1: per-game acceptance proportions paired by game number")


def test_gate_and_holm(res):
    g = res["gate"]
    assert g["n"] == 2 and close(g["mean"], 0.75) and close(g["p"], 0.25) and not g["passed"]
    tests = res["tests"]
    assert not tests["H3"]["tested"] and not tests["H2a C2>C1"]["tested"] and tests["H1a C3>C1"]["tested"]
    assert tests["H3"]["p_holm"] is None
    p1, p2 = tests["H1a C3>C1"]["p"], tests["H1b C1>C2"]["p"]
    lo, hi = sorted([p1, p2])
    assert close(min(tests["H1a C3>C1"]["p_holm"], tests["H1b C1>C2"]["p_holm"]), min(1, 2 * lo))
    assert close(max(tests["H1a C3>C1"]["p_holm"], tests["H1b C1>C2"]["p_holm"]), max(min(1, 2 * lo), hi))
    print("ok gate: ally only; failed gate stops H2/H3 and Holm runs over H1 only")


def test_h4(res):
    t = res["tests"]["H4"]
    # g0: C1 2/3 - 1/3, C2 1 - 1/3, C3 no H4 decisions -> 1/2 ; g1 excluded (C4 g1 lacks the field).
    assert t["n"] == 1 and close(t["mean"], 0.5) and t["p"] is None and "not evaluable" in t["note"]
    assert (t["n_treated_decisions"], t["n_control_decisions"]) == (4, 3), t
    assert t["family"] == "secondary"
    ex = res["h4_exclusions"]
    assert ex == {"could_challenge_proposer": 1, "eligibility_not_logged": 1, "fallback": 0, "system": 1,
                  "games_without_eligibility_field": 1}, ex
    print("ok H4: eligibility filter, unlogged games excluded, C4 counted once, scoped exclusion counts")


def test_holm_and_t():
    assert A.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert A.holm({"a": 0.03, "b": 0.01}) == {"b": 0.02, "a": 0.03}
    assert A.paired_test([0.2])["p"] is None
    # Exact sign-flip on [0.1, 0.3, 0.5]: 8 flips; means >= 0.3 only for (+,+,+) -> 1/8.
    r = A.paired_test([0.1, 0.3, 0.5])
    assert close(r["p"], 1 / 8), r
    # 95% t-interval, df = 2: mean .3 +/- 4.302652730 * .2 / sqrt(3)
    half = 4.302652729911275 * 0.2 / math.sqrt(3)
    assert close(r["ci"][0], 0.3 - half, 1e-9) and close(r["ci"][1], 0.3 + half, 1e-9)
    # Monte Carlo path (2^20 > exact limit): all-positive values must give a small p.
    r = A.paired_test([0.2] * 20)
    assert r["p"] is not None and r["p"] < 0.001, r
    print("ok Holm, exact and Monte Carlo sign-flip, t-interval helpers")


def test_mock_invariants():
    from src.agents import player as player_module
    from src.test.mock_llm import MockLLMClient
    player_module.LLMClient = MockLLMClient
    from src.game.game_scripted_offer import CONDITIONS, GameScriptedOffer
    games = []
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        for g in range(3):
            for c in CONDITIONS:
                games.append(GameScriptedOffer(c, g, seed=g, out_dir=tmp).start_game())
    res = A.compute(games)
    data = res["data"]
    for (c, g), rec in data.games.items():
        if c in A.OFFER_CONDS:
            assert data.offered(c, g) == A.alive_at_offer(rec), "C1-C3 offers go to exactly the agents alive at the offer"
    assert all(r["challenger_can_challenge_proposer"] is not None for r in data.rows)
    for t in res["tests"].values():
        if t["p"] is not None:
            assert 0 <= t["p"] <= 1 and t["ci"][0] <= t["mean"] <= t["ci"][1]
    print(f"ok mock invariants: recipients == agents alive at offer; eligibility logged; {len(data.rows)} rows")


if __name__ == "__main__":
    res = A.compute(synthetic_games())
    test_dataset(res)
    test_h3_shared_control(res)
    test_h2(res)
    test_h1(res)
    test_gate_and_holm(res)
    test_h4(res)
    test_holm_and_t()
    test_mock_invariants()
    print("ALL ANALYSIS CHECKS PASSED")
