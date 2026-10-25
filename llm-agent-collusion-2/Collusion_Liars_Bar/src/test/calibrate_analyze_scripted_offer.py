"""Calibration of analysis/scripted_offer/analyze.py (v7, two-sided) on synthetic records (offline).

Records are generated in the game's own format under the null hypothesis: within a game number
every condition's game is an independent draw from the same distribution (same agents, same seat
after the proposer, same behaviour probabilities), mirroring same seed + unseeded LLM sampling.
The real analysis code is then applied and the script reports, per behavioural test (gate, HA2a,
HA2b, HA2c, HM), the two-sided rejection rate at .05, how often the 95% t-interval covers 0, and
the family-wise error of the HA2 part of Family A (Holm + gate gatekeeping). No judgements are
generated, so HA1a/b/c are not part of the calibration (their values are shares of judged
categories; the same sign-flip test is used). An optional effect shows power.

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/calibrate_analyze_scripted_offer.py [sims] [game numbers]
"""
import contextlib
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import test_analyze_scripted_offer as T

A = T.A
# Calibration only: Monte Carlo permutation p with 1,000 draws everywhere (valid, coarser) to keep
# thousands of simulated analyses fast. The real analysis enumerates exactly when feasible.
A.EXACT_LIMIT = 0
A.PERMUTATIONS = 1000
AGENTS = ["Lily", "Luke", "Quinn"]
SCENARIOS = {
    # (proposer decisions for the seat after the proposer, for other seats, non-challenge prob range,
    #  other-LLM-play decisions per agent, challenge prob range on those, per-game shock sd)
    "moderate": (4.0, 0.4, (0.2, 0.8), 3.0, (0.1, 0.6), 0.10),
    "rare_skewed": (1.5, 0.2, (0.85, 0.98), 1.5, (0.02, 0.15), 0.03),
}
BEHAVIOURAL = ["gate", "HA2a", "HA2b", "HA2c", "HM"]


def poisson(rng, lam):
    k, p, l = 0, 1.0, pow(2.718281828459045, -lam)
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def clip(x):
    return min(0.99, max(0.01, x))


def make_games(rng, sc, G, effect_sel=0.0, effect_hm=0.0, p_dead=1 / 6):
    lam_next, lam_other, ncr, lam_o, ocr, shock = SCENARIOS[sc]
    games = []
    for g in range(G):
        nxt = AGENTS[g % 3]
        base_nc = {a: rng.uniform(*ncr) for a in AGENTS}
        base_oc = {a: rng.uniform(*ocr) for a in AGENTS}
        q_accept = rng.uniform(0.2, 0.8)
        for cond in A.CONDS:
            dead = rng.random() < p_dead     # proposer dead at the offer (no proposer plays, target None)
            offers = {}
            if cond in ("C1", "C2", "C3", "C5"):
                offers = {a: ("ACCEPT" if rng.random() < q_accept else "DECLINE") for a in AGENTS}
            elif cond == "C0":
                offers = {nxt: "ACCEPT"}
            elif cond == "C1b":
                offers = {nxt: ("ACCEPT" if rng.random() < q_accept else "DECLINE")}
            pact = cond in ("C0", "C1", "C2", "C1b")   # C0: the ally is shifted too, so the gate can pass
            decisions = []
            for a in AGENTS:
                s = rng.gauss(0, shock)
                p_nc = clip(base_nc[a] + s + (effect_sel if pact else 0.0))
                if not dead:
                    for _ in range(poisson(rng, lam_next if a == nxt else lam_other)):
                        decisions.append(T.dec(a, "Riley", rng.random() >= p_nc, bluff=rng.random() < 0.5))
                p_oc = clip(base_oc[a] + s + (effect_hm if cond != "C4" else 0.0))
                victim = [b for b in AGENTS if b != a][0]
                for _ in range(poisson(rng, lam_o)):
                    decisions.append(T.dec(a, victim, rng.random() < p_oc))
            games.append(T.game(cond, g, offers, decisions, target=None if dead else nxt,
                                c1b=nxt if cond == "C1b" else None))
    return games


def run(sc, sims, G, effect_sel=0.0, effect_hm=0.0, seed=7):
    rng = random.Random(seed)
    rej = {k: 0 for k in BEHAVIOURAL}
    cover = {k: 0 for k in BEHAVIOURAL}
    evaluable = {k: 0 for k in BEHAVIOURAL}
    fwer = gate_passed = 0
    for _ in range(sims):
        res = A.compute(make_games(rng, sc, G, effect_sel, effect_hm))
        tests = {"gate": res["gate"], **res["tests"]}
        gate_passed += res["gate"]["passed"]
        any_a = False
        for k in BEHAVIOURAL:
            t = tests[k]
            if t["p"] is None:
                continue
            evaluable[k] += 1
            rej[k] += t["p"] < A.ALPHA
            cover[k] += t["ci"][0] <= 0 <= t["ci"][1]
            if k in A.HA2 and t.get("reject"):
                any_a = True
        fwer += any_a
    out = {k: (round(rej[k] / evaluable[k], 3) if evaluable[k] else None,
               round(cover[k] / evaluable[k], 3) if evaluable[k] else None, evaluable[k]) for k in BEHAVIOURAL}
    return out, round(fwer / sims, 3), round(gate_passed / sims, 3)


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for sc in SCENARIOS:
        out, fwer, gp = run(sc, sims, G)
        print(f"NULL {sc} ({sims} sims, {G} game numbers): test -> (two-sided rejection rate at .05, "
              f"95% CI coverage of 0, evaluable sims)")
        for k, v in out.items():
            print(f"   {k:6s} {v}")
        print(f"   gate passed under the null: {gp}; HA2 family-wise error (Holm + gatekeeping): {fwer}")
    for eff in (0.15, 0.30):
        out, _, gp = run("moderate", max(200, sims // 4), G, effect_sel=eff, effect_hm=-eff, seed=11)
        print(f"POWER moderate, {G} game numbers, selectivity effect +{eff} in pact conditions (gate shift equal), "
              f"others-challenge effect -{eff} in message conditions: gate {out['gate'][0]}, HA2a {out['HA2a'][0]}, "
              f"HA2b {out['HA2b'][0]}, HA2c {out['HA2c'][0]}, HM {out['HM'][0]}; gate passed {gp}")
