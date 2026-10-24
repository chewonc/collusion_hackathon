"""Calibration of analysis/scripted_offer/analyze.py on synthetic records (offline, no games run).

Records are generated in the game's own format under the null hypothesis: within a game number
every condition's game is an independent draw from the same distribution (same agents, same
seat after the proposer, same behaviour probabilities), mirroring same seed + unseeded LLM
sampling. The real analysis code is then applied, and the script reports, per test, the
one-sided rejection rate at .05 and how often the 95% t-interval covers the true value 0,
plus the family-wise error of the primary family. An optional effect shows power.

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/calibrate_analyze_scripted_offer.py [sims]
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
G = 12
SCENARIOS = {
    # (proposer decisions for the seat after the proposer, for other seats, non-challenge prob range,
    #  H4 decisions per agent, H4 challenge prob range, per-game shock sd)
    "moderate": (4.0, 0.4, (0.2, 0.8), 3.0, (0.1, 0.6), 0.10),
    "rare_skewed": (1.5, 0.2, (0.85, 0.98), 1.5, (0.02, 0.15), 0.03),
}


def poisson(rng, lam):
    k, p, l = 0, 1.0, pow(2.718281828459045, -lam)
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def clip(x):
    return min(0.99, max(0.01, x))


def make_games(rng, sc, effect_h3=0.0, effect_h4=0.0):
    lam_next, lam_other, ncr, lam_h4, h4r, shock = SCENARIOS[sc]
    games = []
    for g in range(G):
        nxt = AGENTS[g % 3]
        base_nc = {a: rng.uniform(*ncr) for a in AGENTS}
        base_h4 = {a: rng.uniform(*h4r) for a in AGENTS}
        q_accept = rng.uniform(0.2, 0.8)
        for cond in A.CONDS:
            offers = {}
            if cond in A.OFFER_CONDS:
                offers = {a: ("ACCEPT" if rng.random() < q_accept else "DECLINE") for a in AGENTS}
            elif cond == "C0":
                offers = {nxt: "ACCEPT"}
            eff3 = effect_h3 if cond in A.OFFER_CONDS else 0.0
            eff4 = effect_h4 if cond in A.OFFER_CONDS else 0.0
            decisions = []
            for a in AGENTS:
                s = rng.gauss(0, shock)
                p_nc = clip(base_nc[a] + s + eff3)
                for _ in range(poisson(rng, lam_next if a == nxt else lam_other)):
                    decisions.append(T.dec(a, "Riley", rng.random() >= p_nc, bluff=rng.random() < 0.5))
                p_h4 = clip(base_h4[a] + s + eff4)
                victim = [b for b in AGENTS if b != a][0]
                for _ in range(poisson(rng, lam_h4)):
                    decisions.append(T.dec(a, victim, rng.random() < p_h4, elig=False))
            games.append(T.game(cond, g, offers, decisions))
    return games


def run(sc, sims, effect_h3=0.0, effect_h4=0.0, seed=7):
    rng = random.Random(seed)
    names = ["gate"] + A.PRIMARY + ["H4"]
    rej = {k: 0 for k in names}
    cover = {k: 0 for k in names}
    evaluable = {k: 0 for k in names}
    fwer = 0
    for _ in range(sims):
        res = A.compute(make_games(rng, sc, effect_h3, effect_h4))
        tests = {"gate": res["gate"], **res["tests"]}
        any_primary = False
        for k in names:
            t = tests[k]
            if t["p"] is None:
                continue
            evaluable[k] += 1
            rej[k] += t["p"] < A.ALPHA
            cover[k] += t["ci"][0] <= 0 <= t["ci"][1]
            if k in A.PRIMARY and t.get("reject"):
                any_primary = True
        fwer += any_primary
    out = {k: (round(rej[k] / evaluable[k], 3) if evaluable[k] else None,
               round(cover[k] / evaluable[k], 3) if evaluable[k] else None, evaluable[k]) for k in names}
    return out, round(fwer / sims, 3)


if __name__ == "__main__":
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    for sc in SCENARIOS:
        out, fwer = run(sc, sims)
        print(f"NULL {sc} ({sims} sims): test -> (rejection rate at .05, 95% CI coverage of 0, evaluable sims)")
        for k, v in out.items():
            print(f"   {k:10s} {v}")
        print(f"   primary-family FWER (Holm + gatekeeping): {fwer}")
    for eff in (0.10, 0.20):
        out, _ = run("moderate", max(200, sims // 4), effect_h3=eff, effect_h4=eff, seed=11)
        print(f"POWER moderate, effect +{eff} on H3 non-challenge and H4 challenge rates, 12 game numbers: "
              f"H3 {out['H3'][0]}, H4 {out['H4'][0]}")
