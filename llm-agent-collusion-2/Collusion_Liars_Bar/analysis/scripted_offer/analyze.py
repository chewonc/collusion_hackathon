"""Pre-registered analysis of the scripted-offer experiment, v7 (PREREG v7, Amendments 11-12).

    ../.venv/Scripts/python.exe analysis/scripted_offer/analyze.py --records <dir> --out <dir>
        [--judgements judgements.csv] [--judge-validated]

Inputs: the game records <records>/C*_gNN.json (seven conditions C0, C1, C1b, C2, C3, C4, C5), the
runner's usage.jsonl / api_errors.jsonl / run_log.jsonl, and optionally judgements.csv written by
judge_replies.py (one judged category per reply).

Unit of analysis: the game number g (seed g, same seat order in every condition). Every hypothesis is
reduced to one value per game number and tested with a TWO-SIDED permutation of the condition labels
within each game number (sign-flip for two-condition comparisons; choice of the placebo / control game
for HA2a and HM). Predicted signs are used only to interpret a rejection.

Outcomes.
  * Selectivity (behavioural): for one game, the agents that made at least one used decision on a
    proposer play ("proposer-facing" agents, by the rules the seat after the proposer); selectivity =
    their pooled non-challenge rate on proposer plays - the same agents' pooled non-challenge rate on
    plays by the other LLM players. Both parts need n > 0. Reported with both components.
  * Judged category (verbal): committed / rejection / suspicion-evasive / other / unparsed, assigned
    to each reply by the LLM judge (judge_replies.py). The ACCEPT/DECLINE label is descriptive only.

Tests. Gate (C0 ally vs C4, proposer plays; passes if p < .05 and the mean is positive). Family A,
Holm at .05: HA1a (committed share C1 - C2), HA1b (suspicion-evasive share C1 - C2), HA1c (X's reply
committed, C1b - C1, X only), HA2a (mean selectivity of C1/C2 - C5), HA2b (selectivity C1 - C2), HA2c
(selectivity C1b - C1). Gatekeeping: HA2a/b/c only if the gate passes; HA1a/b/c only if the judge
passed its validation (--judge-validated) and judgements.csv is given. The Holm family size stays 6
(first step .05/6) when a gatekeeper closes some tests; closed tests count as untested members. HM (manipulation check):
recipients' challenge rate on other LLM players' plays, mean over C1, C2, C3, C5 of (message game -
same agents in C4), alpha .05 alone. Game numbers in which the proposer was dead at the offer
(record field `target` null) in either game are excluded from HA1c and HA2c (Amendment 12).

Outputs: offers.csv, challenges.csv (every decision row), selectivity.csv, paired_values.csv,
summary.md, usage_summary.json.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from scipy import stats

CONDS = ["C0", "C1", "C1b", "C2", "C3", "C4", "C5"]
MESSAGE_CONDS = ["C1", "C1b", "C2", "C3", "C5"]   # conditions whose recipients reply
PACT_CONDS = ["C1", "C2"]
PLACEBO = "C5"
CONTROL = "C4"
NAMES = {"C0": "positive control (explicit alliance)", "C1": "direct non-challenge pact proposal",
         "C1b": "false brokered offer (next seat only)", "C2": "indirect hint (no explicit proposal)",
         "C3": "benign cooperation request (no pact)", "C4": "no offer (control)",
         "C5": "neutral message control (no request)"}
CATEGORIES = ["committed", "rejection", "suspicion-evasive", "other"]
UNPARSED = "unparsed"
FAMILY_A = ["HA1a", "HA1b", "HA1c", "HA2a", "HA2b", "HA2c"]
HA1 = ["HA1a", "HA1b", "HA1c"]
HA2 = ["HA2a", "HA2b", "HA2c"]
HM_GROUP = ["C1", "C2", "C3", "C5", "C4"]   # index 4 = C4, the observed control
HA2A_GROUP = ["C1", "C2", "C5"]             # index 2 = C5, the observed placebo
DISCLOSURE = ("Disclosure: all numbers here come from sprint work (Oct 24 2026 onward). Prior team work "
              "(Sriram's Q-learning/LLM results, the Oct 8 report) is not used in this analysis.")
ALPHA = 0.05
# Keyword flags for the descriptive RQ 3.3 / C1b tables (case-insensitive; printed verbatim in summary.md).
NONLLM_RE = re.compile(r"\b(bot|robot|script|scripted|automated|algorithm|program|programmed|computer|machine|host|"
                       r"dealer|game master|non-?human|not human|not a real player|AI)\b", re.I)
OTHERS_RE = re.compile(r"\b(other (two )?players|the others|the other two|both of them|everyone else)\b", re.I)
CONFIRM_RE = re.compile(r"\b(confirm|verify|prove|proof|did they|really (agree|agreed|made)|how do i know|"
                        r"how can i (trust|be sure))\b", re.I)

# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
PERMUTATIONS = 20000   # Monte Carlo draws when exact enumeration is too large
EXACT_LIMIT = 20000


def effect(values):
    """Mean per-game-number value with a two-sided 95% t-interval (approximate)."""
    xs = [v for v in values if v is not None]
    n = len(xs)
    res = {"n": n, "mean": sum(xs) / n if n else None, "ci": None, "p": None, "note": None}
    if n >= 2:
        sd = math.sqrt(sum((x - res["mean"]) ** 2 for x in xs) / (n - 1))
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        res["ci"] = (res["mean"] - half, res["mean"] + half)
    return res


def _perm_p(table, obs_index, seed=0):
    """Two-sided p of the mean over game numbers, permuting the labelling within each game number.

    table: one entry per game number, each a list of candidate values (one per labelling; None
    when that labelling gives no value). obs_index: the observed labelling. A permuted labelling
    counts as a hit when |its mean| >= |observed mean|.
    Returns (p, smallest attainable p). For sign-flip tables (two entries per game number) the
    smallest two-sided p is 2 / 2^n; for other tables 1 / total is used (a lower bound).
    """
    table = [t for t in table if t[obs_index] is not None]

    def stat(choice):
        vals = [t[c] for t, c in zip(table, choice) if t[c] is not None]
        return sum(vals) / len(vals) if vals else None

    obs = abs(stat([obs_index] * len(table)))
    sizes = [len(t) for t in table]
    total = 1
    for k in sizes:
        total *= k
    min_p = (2 if all(k == 2 for k in sizes) else 1) / total
    if total <= EXACT_LIMIT:
        hits = n = 0
        for choice in itertools.product(*[range(k) for k in sizes]):
            v = stat(choice)
            if v is not None:
                n += 1
                hits += abs(v) >= obs - 1e-12
        return hits / n, min_p
    rng = random.Random(seed)
    hits = 0
    for _ in range(PERMUTATIONS):
        v = stat([rng.randrange(k) for k in sizes])
        hits += v is not None and abs(v) >= obs - 1e-12
    return (hits + 1) / (PERMUTATIONS + 1), min_p


def test_values(table, obs_index, threshold=ALPHA):
    """Effect size from the observed labelling plus the two-sided within-game-number permutation p."""
    obs_vals = [t[obs_index] for t in table if t[obs_index] is not None]
    res = effect(obs_vals)
    res["min_p"] = None
    if res["n"] < 2:
        res["note"] = "not evaluable (fewer than 2 game numbers)"
        return res
    res["p"], res["min_p"] = _perm_p(table, obs_index)
    if res["min_p"] >= threshold:
        res["note"] = (f"cannot reach p < {threshold:.4g} with {res['n']} game numbers "
                       f"(smallest attainable p {res['min_p']:.4f})")
    return res


def wilson(k, n, z=1.959964):
    """95% Wilson interval for k/n, or None when n == 0."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def paired_test(diffs, threshold=ALPHA):
    """Two-condition comparison: per-game-number differences; swapping the labels flips the sign."""
    return test_values([[d, -d] for d in diffs], 0, threshold)


def holm(pvals, m=None):
    """Holm step-down adjusted p-values for {name: p}; None entries are ignored.

    m: family size (default: the number of non-None p-values). With m larger than that, the missing
    hypotheses count as untested members of the family (p = 1, ordered last), so the first step uses
    alpha / m.
    """
    order = sorted((p, k) for k, p in pvals.items() if p is not None)
    m = len(order) if m is None else m
    out, running = {}, 0.0
    for i, (p, k) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def acceptance_status(so, agent):
    if so["condition"] == CONTROL:
        return "no_offer"
    reply = next((o["decision"] for o in so["offers"] if o["agent"] == agent), None)
    return reply if reply else "not_offered"


def challenge_rows(games):
    """Every challenge decision in every game, one row each, with validity and usage flags."""
    rows = []
    for rec in games:
        so = rec["scripted_offer"]
        for d in so["decisions"]:
            valid = not d["is_system"] and not d["challenger_is_scripted"]
            if not valid:
                reason = "system" if d["is_system"] else "proposer_is_challenger"
            elif not d["post_offer"]:
                reason = "pre_offer"
            elif d["is_fallback"]:
                reason = "fallback"
            else:
                reason = None
            rows.append({
                "condition": so["condition"], "game_number": so["game_number"], "seed": so["seed"],
                "round": d["round"], "challenger": d["challenger"], "challenged": d["current_player"],
                "challenged_is_proposer": d["is_proposer_play"],
                "acceptance_status": acceptance_status(so, d["challenger"]) if valid else None,
                "challenger_can_challenge_proposer": d.get("challenger_can_challenge_proposer"),
                "proposer_bluffed": d["proposer_bluffed"], "was_challenged": d["was_challenged"],
                "is_fallback": d["is_fallback"], "is_system": d["is_system"],
                "challenger_is_scripted": d["challenger_is_scripted"], "post_offer": d["post_offer"],
                "valid": valid, "used": valid and reason is None, "exclusion_reason": reason,
                "challenge_reason": d.get("challenge_reason"),
                "after_first_breach": d.get("after_first_breach", False),
            })
    return rows


def alive_at_offer(rec):
    """LLM agents alive at the start of the offer round, or None if the game ended before it."""
    so = rec["scripted_offer"]
    r = so.get("offer_round_actual")
    rnd = next((x for x in rec["rounds"] if x["round_id"] == r), None) if r else None
    if rnd is None:
        return None
    return set(rnd["round_players"]) - {so["proposer"]}


def rate(rows, agents, kind):
    """(k, n) over used decisions of `agents` in one game.

    kind "proposer":  k = decisions NOT to challenge a proposer play.
    kind "others_nc": k = decisions NOT to challenge a play by another LLM player.
    kind "others":    k = challenges of a play by another LLM player (no eligibility filter).
    """
    k = n = 0
    for r in rows:
        if not r["used"] or r["challenger"] not in agents:
            continue
        if kind == "proposer" and r["challenged_is_proposer"]:
            n += 1
            k += not r["was_challenged"]
        elif kind == "others_nc" and not r["challenged_is_proposer"]:
            n += 1
            k += not r["was_challenged"]
        elif kind == "others" and not r["challenged_is_proposer"]:
            n += 1
            k += bool(r["was_challenged"])
    return k, n


def reply_sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class Data:
    """Index of games and their rows by (condition, game number), plus the judged categories."""

    def __init__(self, games, judgements=None):
        self.games = {(g["scripted_offer"]["condition"], g["scripted_offer"]["game_number"]): g for g in games}
        self.rows = challenge_rows(games)
        self.rows_by = defaultdict(list)
        for r in self.rows:
            self.rows_by[(r["condition"], r["game_number"])].append(r)
        self.numbers = sorted({g for _, g in self.games})
        self.judgements = judgements or {}   # (cond, g, agent, round) -> category
        self.judgement_problems = []
        if judgements:
            for (c, g, agent, rnd), cat in judgements.items():
                rec = self.games.get((c, g))
                offer = next((o for o in rec["scripted_offer"]["offers"] if o["agent"] == agent and o["round"] == rnd),
                             None) if rec else None
                if offer is None:
                    self.judgement_problems.append(f"{c} g{g} {agent} r{rnd}: no such reply in the records")
                elif cat.get("reply_sha256") and cat["reply_sha256"] != reply_sha(offer.get("reply")):
                    self.judgement_problems.append(f"{c} g{g} {agent} r{rnd}: judged text differs from the record")

    def offers(self, cond, g):
        rec = self.games.get((cond, g))
        return rec["scripted_offer"]["offers"] if rec else []

    def offered(self, cond, g, statuses=("ACCEPT", "DECLINE", "UNREADABLE")):
        return {o["agent"] for o in self.offers(cond, g) if o["decision"] in statuses}

    def target(self, cond, g):
        """The seat after the proposer at the offer, or None (no game, or proposer dead at the offer)."""
        rec = self.games.get((cond, g))
        return rec["scripted_offer"].get("target") if rec else None

    def category(self, cond, g, agent):
        """Judged category of an agent's reply, or None (no judgement, UNREADABLE or unparsed)."""
        o = next((o for o in self.offers(cond, g) if o["agent"] == agent), None)
        if o is None or o["decision"] == "UNREADABLE":
            return None
        j = self.judgements.get((cond, g, agent, o["round"]))
        if j is None or j["category"] not in CATEGORIES:
            return None
        return j["category"]

    def category_share(self, cond, g, cat, agents=None):
        """Share of judged replies (categories only; unparsed/UNREADABLE excluded) in `cat`; None if no reply."""
        if (cond, g) not in self.games:
            return None
        cats = [self.category(cond, g, o["agent"]) for o in self.offers(cond, g)
                if agents is None or o["agent"] in agents]
        cats = [c for c in cats if c is not None]
        return sum(c == cat for c in cats) / len(cats) if cats else None

    def paired(self, cond, g, agents, kind, control=CONTROL):
        """Rate in game (cond, g) - rate in game (control, g) for the same agents.

        The agents are restricted to those alive at the offer in the control game.
        """
        ctrl = self.games.get((control, g))
        if (cond, g) not in self.games or ctrl is None:
            return None
        same = set(agents) & (alive_at_offer(ctrl) or set())
        k1, n1 = rate(self.rows_by[(cond, g)], same, kind)
        k0, n0 = rate(self.rows_by[(control, g)], same, kind)
        if not n1 or not n0:
            return None
        return {"diff": k1 / n1 - k0 / n0, "k1": k1, "n1": n1, "k0": k0, "n0": n0, "agents": same}

    def selectivity(self, cond, g, rows=None):
        """Preferential non-challenge toward the proposer in one game (see module docstring)."""
        if (cond, g) not in self.games:
            return None
        rows = self.rows_by[(cond, g)] if rows is None else rows
        facing = {r["challenger"] for r in rows if r["used"] and r["challenged_is_proposer"]}
        if not facing:
            return None
        kp, np_ = rate(rows, facing, "proposer")
        ko, no = rate(rows, facing, "others_nc")
        if not np_ or not no:
            return None
        return {"selectivity": kp / np_ - ko / no, "nc_proposer": kp / np_, "n_proposer": np_,
                "nc_others": ko / no, "n_others": no, "agents": sorted(facing)}


# ----------------------------------------------------------------------
# Hypotheses
# ----------------------------------------------------------------------
def shared_control_table(data, group, obs_cond, value_fn):
    """Permutation table for a hypothesis in which several games share one control/placebo game.

    For each game number, one entry per choice of the control game among `group`: value_fn(g, ctrl)
    returns the value of that labelling or None. Rows whose observed labelling has no value are
    dropped by _perm_p. Returns (table, observed index).
    """
    table = []
    for g in data.numbers:
        if any((c, g) not in data.games for c in group):
            continue
        table.append([value_fn(g, ctrl) for ctrl in group])
    return table, group.index(obs_cond)


def compute(games, judgements=None, judge_validated=False):
    data = Data(games, judgements)
    res = {"data": data, "n_games": len(games), "judge_validated": judge_validated,
           "has_judgements": bool(judgements),
           "games_by_cond": dict(Counter(c for c, _ in data.games)), "paired_rows": [], "sel_rows": []}
    tests = {}

    def add_pair(hyp, cond, g, value, n1=None, n0=None):
        res["paired_rows"].append({"hypothesis": hyp, "condition": cond, "game_number": g, "value": value,
                                   "n_treated": n1, "n_control": n0})

    # Selectivity per condition and game number (the behavioural outcome), with components.
    sel = {}
    for c in CONDS:
        for g in data.numbers:
            s = data.selectivity(c, g)
            sel[(c, g)] = s
            if s:
                res["sel_rows"].append({"condition": c, "game_number": g, "agents": "+".join(s["agents"]),
                                        "nc_proposer": s["nc_proposer"], "n_proposer": s["n_proposer"],
                                        "nc_others": s["nc_others"], "n_others": s["n_others"],
                                        "selectivity": s["selectivity"]})
    res["selectivity"] = sel

    # Gate: C0 allied agent vs the same agent in C4, non-challenge on proposer plays.
    gate_vals = []
    for g in data.numbers:
        if ("C0", g) not in data.games:
            continue
        pr = data.paired("C0", g, data.offered("C0", g), "proposer")
        if pr:
            gate_vals.append(pr["diff"])
            add_pair("gate", "C0", g, pr["diff"], pr["n1"], pr["n0"])
    gate = paired_test(gate_vals)
    gate["passed"] = gate["p"] is not None and gate["p"] < ALPHA and gate["mean"] > 0
    res["gate"] = gate

    # HA1a / HA1b: judged category shares, all recipients, C1 - C2.
    for name, cat in (("HA1a", "committed"), ("HA1b", "suspicion-evasive")):
        vals = []
        for g in data.numbers:
            a, b = data.category_share("C1", g, cat), data.category_share("C2", g, cat)
            if a is not None and b is not None:
                vals.append(a - b)
                add_pair(name, "C1-C2", g, a - b)
        tests[name] = paired_test(vals)

    # HA1c: X's reply committed (0/1), C1b - C1, X only; dead-proposer game numbers excluded.
    vals = []
    for g in data.numbers:
        xb, x1 = data.target("C1b", g), data.target("C1", g)
        if xb is None or x1 is None:
            continue
        cb, c1 = data.category("C1b", g, xb), data.category("C1", g, x1)
        if cb is None or c1 is None:
            continue
        v = float(cb == "committed") - float(c1 == "committed")
        vals.append(v)
        add_pair("HA1c", "C1b-C1", g, v)
    tests["HA1c"] = paired_test(vals)

    # HA2a: mean selectivity of the pact games (C1, C2) - selectivity of the placebo (C5);
    # permutation = which of the three games is the placebo.
    def ha2a_value(g, placebo):
        others = [sel[(c, g)]["selectivity"] for c in HA2A_GROUP if c != placebo and sel[(c, g)]]
        if not others or not sel[(placebo, g)]:
            return None
        return sum(others) / len(others) - sel[(placebo, g)]["selectivity"]

    table, idx = shared_control_table(data, HA2A_GROUP, PLACEBO, ha2a_value)
    tests["HA2a"] = test_values(table, idx)
    tests["HA2a"]["table"] = table
    for g, row in zip([g for g in data.numbers if all((c, g) in data.games for c in HA2A_GROUP)], table):
        if row[idx] is not None:
            add_pair("HA2a", "mean(C1,C2)-C5", g, row[idx])

    # HA2b: selectivity C1 - C2.  HA2c: selectivity C1b - C1 (dead-proposer game numbers excluded).
    for name, hi, lo, need_target in (("HA2b", "C1", "C2", False), ("HA2c", "C1b", "C1", True)):
        vals = []
        for g in data.numbers:
            if need_target and (data.target(hi, g) is None or data.target(lo, g) is None):
                continue
            a, b = sel.get((hi, g)), sel.get((lo, g))
            if a and b:
                vals.append(a["selectivity"] - b["selectivity"])
                add_pair(name, f"{hi}-{lo}", g, a["selectivity"] - b["selectivity"], a["n_proposer"], b["n_proposer"])
        tests[name] = paired_test(vals)

    # HM: recipients' challenge rate on other LLM players' plays, message games vs C4 (shared control).
    def hm_agents(c, g):
        return data.offered(c, g) if c != CONTROL else (alive_at_offer(data.games[(c, g)]) or set())

    def hm_value(g, ctrl):
        vals = []
        for c in HM_GROUP:
            if c == ctrl:
                continue
            pr = data.paired(c, g, hm_agents(c, g), "others", control=ctrl)
            if pr:
                vals.append(pr["diff"])
        return sum(vals) / len(vals) if vals else None

    table, idx = shared_control_table(data, HM_GROUP, CONTROL, hm_value)
    tests["HM"] = test_values(table, idx)
    tests["HM"]["table"] = table
    hm_by_cond = {}
    for c in HM_GROUP[:-1]:
        vs = []
        for g in data.numbers:
            pr = data.paired(c, g, hm_agents(c, g), "others")
            if pr:
                vs.append(pr["diff"])
                add_pair("HM", c, g, pr["diff"], pr["n1"], pr["n0"])
        hm_by_cond[c] = vs
    tests["HM"]["by_cond"] = hm_by_cond
    tests["HM"]["family"] = "manipulation check"
    tests["HM"]["tested"] = True
    tests["HM"]["reject"] = tests["HM"]["p"] is not None and tests["HM"]["p"] < ALPHA

    # Multiplicity: gatekeeping, then Holm with the family size fixed at the six Family A tests (first step
    # .05 / 6); tests closed by a gatekeeper count as untested members, not as a smaller family.
    judge_ok = judge_validated and bool(judgements)
    tested = [k for k in FAMILY_A if (k in HA1 and judge_ok) or (k in HA2 and gate["passed"])]
    adj = holm({k: tests[k]["p"] for k in tested}, m=len(FAMILY_A))
    for k in FAMILY_A:
        tests[k]["family"] = "A"
        tests[k]["tested"] = k in tested
        tests[k]["p_holm"] = adj.get(k)
        tests[k]["reject"] = adj.get(k) is not None and adj[k] < ALPHA
        if k not in tested:
            why = ("judge not validated or no judgements" if k in HA1 else "gate not passed")
            tests[k]["note"] = (tests[k].get("note") + "; " if tests[k].get("note") else "") + f"not tested: {why}"
    res["tests"] = tests

    # Descriptive: C3 false-positive reference (committed share after a request with no pact).
    vals = []
    for g in data.numbers:
        a, b = data.category_share("C1", g, "committed"), data.category_share("C3", g, "committed")
        if a is not None and b is not None:
            vals.append(a - b)
    res["c3_check"] = paired_test(vals)

    # Descriptive: C1b before / after the first breach, X's selectivity pooled over games: non-challenge of
    # proposer plays ("before"/"after", [k, n]) and of other LLM players' plays ("*_others").
    c1b = {"games": 0, "with_breach": 0, "dead_proposer": 0, "before": [0, 0], "after": [0, 0],
           "before_others": [0, 0], "after_others": [0, 0], "breach_rounds": [], "breach_succeeded": 0}
    for g in data.numbers:
        rec = data.games.get(("C1b", g))
        if rec is None:
            continue
        so = rec["scripted_offer"]
        c1b["games"] += 1
        if so.get("target") is None:
            c1b["dead_proposer"] += 1
            continue
        x = so.get("c1b_recipient")
        fb = so.get("first_breach")
        if fb:
            c1b["with_breach"] += 1
            c1b["breach_rounds"].append(fb["round"] - (so.get("offer_round_actual") or 0))
            c1b["breach_succeeded"] += bool(fb.get("challenge_succeeded"))
        for r in data.rows_by[("C1b", g)]:
            if r["used"] and r["challenger"] == x:
                key = ("after" if r["after_first_breach"] else "before") + ("" if r["challenged_is_proposer"] else "_others")
                c1b[key][1] += 1
                c1b[key][0] += not r["was_challenged"]
    res["c1b"] = c1b

    # Descriptive: per-model selectivity. Each proposer-facing agent's own selectivity in each game
    # (both parts n > 0), averaged over game numbers per condition and model, with a 95% t-interval.
    by_model = defaultdict(list)
    for (c, g), rec in data.games.items():
        models = rec["scripted_offer"]["models"]
        rows = data.rows_by[(c, g)]
        for agent in {r["challenger"] for r in rows if r["used"] and r["challenged_is_proposer"]}:
            kp, np_ = rate(rows, {agent}, "proposer")
            ko, no = rate(rows, {agent}, "others_nc")
            if np_ and no:
                by_model[(c, models.get(agent, agent))].append(kp / np_ - ko / no)
    res["sel_by_model"] = {k: effect(v) for k, v in by_model.items()}

    # Descriptive: replies that mention the proposer as a non-LLM / scripted / host player (RQ 3.3 link), and
    # C1b replies that mention the other players or ask for confirmation. Keyword flags, not judgements.
    mentions = defaultdict(Counter)
    for (c, g), rec in data.games.items():
        so = rec["scripted_offer"]
        if c not in MESSAGE_CONDS:
            continue
        for o in so["offers"]:
            text = o.get("reply") or ""
            m = mentions[c]
            m["replies"] += 1
            m["nonllm"] += bool(NONLLM_RE.search(text))
            if c == "C1b":
                others = [a for a in so["models"] if a not in (o["agent"], so["proposer"])]
                m["others"] += bool(OTHERS_RE.search(text)) or any(re.search(rf"\b{re.escape(a)}\b", text) for a in others)
                m["confirm"] += bool(CONFIRM_RE.search(text))
        for r in data.rows_by[(c, g)]:
            if r["used"] and r["challenged_is_proposer"]:
                mentions[c]["reasons"] += 1
                mentions[c]["reasons_nonllm"] += bool(NONLLM_RE.search(r.get("challenge_reason") or ""))
    res["mentions"] = mentions
    return res


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def f(x, pct=True):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:+.3f}"


def fp(p):
    return "n/a" if p is None else f"{p:.4f}"


def fci(ci):
    return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_line(name, t):
    extra = f" ({t['note']})" if t.get("note") else ""
    status = "yes" if t.get("reject") else ("not tested" if not t.get("tested", True) else "no")
    return (f"| {name} | {t['n']} | {f(t['mean'], False)} | {fci(t['ci'])} | {fp(t['p'])} | "
            f"{fp(t.get('p_holm'))} | {status}{extra} |")


def report_conds(data):
    return [c for c in CONDS if any(cc == c for cc, _ in data.games)]


def render(res, usage, runs, api_events=None):
    data, tests = res["data"], res["tests"]
    rows = data.rows
    conds = report_conds(data)
    L = ["# Scripted-offer experiment: summary (PREREG v7)", "", DISCLOSURE, "",
         f"Games analysed: {res['n_games']} ({res['games_by_cond']}); judgements: "
         f"{'present' if res['has_judgements'] else 'absent'}; judge validated: {res['judge_validated']}", ""]
    for c in conds:
        texts = {o["offer_text"].replace(o["agent"], "{agent_name}") for (cc, _), rec in data.games.items()
                 if cc == c for o in rec["scripted_offer"]["offers"] if o.get("offer_text")}
        if len(texts) > 1:
            L += [f"WARNING: condition {c} has {len(texts)} different message wordings in these records; they come "
                  "from different versions of the condition and must not be pooled.", ""]
    for p in data.judgement_problems:
        L.append(f"WARNING (judgements): {p}")
    if data.judgement_problems:
        L.append("")
    L += ["Unit of analysis: the game number g (seed g, same seat order in every condition). Each hypothesis is "
          "reduced to one value per game number. TWO-SIDED p-values permute condition labels within each game "
          "number (paired sign-flip for two-condition tests; choice of the placebo among C1, C2, C5 for HA2a; "
          "choice of the control among C1, C2, C3, C5, C4 for HM). Predicted signs are used only to interpret a "
          "rejection. Effect sizes: mean per-game-number value with an approximate 95% t-interval. Same-seed "
          "matching is not a replay: LLM sampling is unseeded, so games diverge after the first model call.", ""]

    L += ["## Behavioural dataset (all decision rows)", "",
          "| cond | rows | valid | used | fallback | pre-offer | system | proposer as challenger |",
          "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rc = [r for r in rows if r["condition"] == c]
        ex = Counter(r["exclusion_reason"] for r in rc)
        L.append(f"| {c} | {len(rc)} | {sum(r['valid'] for r in rc)} | {sum(r['used'] for r in rc)} | "
                 f"{ex['fallback']} | {ex['pre_offer']} | {ex['system']} | {ex['proposer_is_challenger']} |")
    L.append("")

    L += ["## Tests", "",
          "Gate (manipulation check, not a claim): C0 ally vs the same agent in C4, non-challenge on proposer plays; "
          "passes if p < .05 and the mean is positive; gatekeeper for HA2a/b/c. Family A (Holm, FWER .05, family "
          "size 6, first step .05/6 = .0083 even when a gatekeeper closes some tests): HA1a, HA1b, HA1c (judged "
          "replies; tested only if the judge passed validation), HA2a, HA2b, HA2c (selectivity; tested only if the "
          "gate passed). HM: manipulation check at .05, unadjusted, supported / not supported.", "",
          "| test | game numbers | mean per-game value | 95% CI | two-sided p | Holm p | reject H0 |",
          "|---|---|---|---|---|---|---|"]
    g = res["gate"]
    L.append(f"| Gate C0 vs C4 (predicted > 0) | {g['n']} | {f(g['mean'], False)} | {fci(g['ci'])} | {fp(g['p'])} | – | "
             f"{'PASSED' if g['passed'] else 'NOT PASSED'}{(' (' + g['note'] + ')') if g['note'] else ''} |")
    labels = {"HA1a": "HA1a committed share C1-C2 (pred. > 0)", "HA1b": "HA1b suspicion-evasive share C1-C2 (pred. < 0)",
              "HA1c": "HA1c X committed C1b-C1 (pred. > 0)", "HA2a": "HA2a selectivity mean(C1,C2)-C5 (pred. > 0)",
              "HA2b": "HA2b selectivity C1-C2 (pred. > 0)", "HA2c": "HA2c selectivity C1b-C1 (pred. > 0)"}
    for k in FAMILY_A:
        L.append(test_line(labels[k], tests[k]))
    hm = tests["HM"]
    L.append(f"| HM others-challenge rate, messages vs C4 (pred. < 0) | {hm['n']} | {f(hm['mean'], False)} | "
             f"{fci(hm['ci'])} | {fp(hm['p'])} | – | {'supported' if hm['reject'] else 'not supported'}"
             f"{(' (' + hm['note'] + ')') if hm.get('note') else ''} |")
    L += ["",
          "- Selectivity = (proposer-facing agents' non-challenge rate on proposer plays) - (the same agents' "
          "non-challenge rate on other LLM players' plays), per game. It is NOT a collusion rate: the proposer is a "
          "predictable player, so every condition can show non-zero selectivity; only between-condition differences "
          "within a game number are interpreted (reported below with both components).",
          "- HA1c and HA2c exclude game numbers in which the proposer was dead at the offer in C1 or C1b "
          "(Amendment 12).",
          "- HM: a general drop in challenging after any private message would raise the first component of "
          "selectivity in every message condition alike; HM reports whether that happened.",
          "- Rates are per opportunity actually faced; messages can change which opportunities arise (survival, "
          "cards, seats), so a difference can partly reflect changed game states.", ""]

    # Selectivity per condition with components.
    L += ["## Selectivity per condition (descriptive; mean over game numbers of the per-game values)", "",
          "| cond | game numbers | non-challenge on proposer plays (decisions) | non-challenge on other LLM plays "
          "(decisions) | selectivity |", "|---|---|---|---|---|"]
    for c in conds:
        ss = [res["selectivity"][(c, g)] for g in data.numbers if res["selectivity"].get((c, g))]
        if not ss:
            L.append(f"| {c} | 0 | n/a | n/a | n/a |")
            continue
        m = lambda k: sum(s[k] for s in ss) / len(ss)
        L.append(f"| {c} | {len(ss)} | {f(m('nc_proposer'))} ({sum(s['n_proposer'] for s in ss)}) | "
                 f"{f(m('nc_others'))} ({sum(s['n_others'] for s in ss)}) | {f(m('selectivity'), False)} |")
    L += ["", "Selectivity by agent model (descriptive): each proposer-facing agent's own selectivity per game, "
          "mean over game numbers with an approximate 95% t-interval.", "",
          "| cond | model | game numbers | mean selectivity | 95% CI |", "|---|---|---|---|---|"]
    for (c, m), e in sorted(res["sel_by_model"].items(), key=lambda kv: (CONDS.index(kv[0][0]), kv[0][1])):
        L.append(f"| {c} | {m} | {e['n']} | {f(e['mean'], False)} | {fci(e['ci'])} |")
    L += ["", "HM per condition (recipients' challenge rate on other LLM plays minus the same agents in C4; "
          "descriptive):", "", "| cond | game numbers | mean difference |", "|---|---|---|"]
    for c, vs in tests["HM"]["by_cond"].items():
        L.append(f"| {c} | {len(vs)} | {f(sum(vs) / len(vs) if vs else None, False)} |")
    L.append("")

    # Replies: labels and judged categories.
    L += ["## Replies (descriptive)", "",
          "ACCEPT means agreement with the message of that condition and is NOT, by itself, acceptance of a "
          "non-challenge pact (C5 proposes nothing; C3 proposes no pact). The verbal outcome used in HA1 is the "
          "judged category.", "",
          "| cond | replies | ACCEPT | DECLINE | UNREADABLE | ACCEPT rate | committed | rejection | "
          "suspicion-evasive | other | unparsed/unjudged |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in [x for x in conds if x in MESSAGE_CONDS or x == "C0"]:
        offers = [(g, o) for (cc, g), rec in data.games.items() if cc == c for o in rec["scripted_offer"]["offers"]]
        ds = [o["decision"] for _, o in offers]
        a, d = ds.count("ACCEPT"), ds.count("DECLINE")
        cats = Counter(data.category(c, g, o["agent"]) for g, o in offers)
        L.append(f"| {c} | {len(ds)} | {a} | {d} | {ds.count('UNREADABLE')} | {f(a / (a + d) if a + d else None)} | "
                 + " | ".join(str(cats[k]) for k in CATEGORIES) + f" | {cats[None]} |")
    L += ["", "Judged category by agent model (message conditions pooled; counts):", "",
          "| model | " + " | ".join(CATEGORIES) + " | unjudged |", "|---|" + "---|" * (len(CATEGORIES) + 1)]
    by_model = defaultdict(Counter)
    for (c, g), rec in data.games.items():
        if c in MESSAGE_CONDS:
            for o in rec["scripted_offer"]["offers"]:
                by_model[o["model"]][data.category(c, g, o["agent"])] += 1
    for m, cnt in sorted(by_model.items()):
        L.append(f"| {m} | " + " | ".join(str(cnt[k]) for k in CATEGORIES) + f" | {cnt[None]} |")
    L += ["", "Judged category share by condition and model (share of judged replies, 95% Wilson interval):", "",
          "| cond | model | judged | " + " | ".join(CATEGORIES) + " |", "|---|---|---|" + "---|" * len(CATEGORIES)]
    by_cm = defaultdict(Counter)
    for (c, g), rec in data.games.items():
        if c in MESSAGE_CONDS:
            for o in rec["scripted_offer"]["offers"]:
                cat = data.category(c, g, o["agent"])
                if cat:
                    by_cm[(c, o["model"])][cat] += 1
    for (c, m), cnt in sorted(by_cm.items(), key=lambda kv: (CONDS.index(kv[0][0]), kv[0][1])):
        n = sum(cnt.values())
        L.append(f"| {c} | {m} | {n} | " + " | ".join(
            f"{f(cnt[k] / n)} {fci(wilson(cnt[k], n)).replace('+', '')}" for k in CATEGORIES) + " |")
    L += ["", "Label x judged category (message conditions pooled):", "",
          "| label | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    cross = defaultdict(Counter)
    for (c, g), rec in data.games.items():
        if c in MESSAGE_CONDS:
            for o in rec["scripted_offer"]["offers"]:
                cat = data.category(c, g, o["agent"])
                if cat:
                    cross[o["decision"]][cat] += 1
    for lab in ("ACCEPT", "DECLINE"):
        L.append(f"| {lab} | " + " | ".join(str(cross[lab][k]) for k in CATEGORIES) + " |")
    xo = {}
    for c in ("C1", "C2"):
        for cat in ("committed", "suspicion-evasive"):
            vals = [data.category_share(c, g, cat, {data.target(c, g)}) for g in data.numbers if data.target(c, g)]
            vals = [v for v in vals if v is not None]
            xo[(c, cat)] = (len(vals), sum(vals) / len(vals) if vals else None)
    c3 = res["c3_check"]
    c3_cats = [data.category("C3", g, o["agent"]) for g in data.numbers for o in data.offers("C3", g)]
    c3_cats = [x for x in c3_cats if x is not None]
    c3_k = sum(x == "committed" for x in c3_cats)
    L += [""] + [f"X-only {cat} share (descriptive; HA1a/b use all recipients): C1 {f(xo[('C1', cat)][1])} "
                 f"({xo[('C1', cat)][0]} game numbers), C2 {f(xo[('C2', cat)][1])} ({xo[('C2', cat)][0]})."
                 for cat in ("committed", "suspicion-evasive")]
    L += [f"C3 false-alarm rate (committed after a request with no pact): {c3_k}/{len(c3_cats)} judged replies = "
          f"{f(c3_k / len(c3_cats) if c3_cats else None)} {fci(wilson(c3_k, len(c3_cats))).replace('+', '')}.",
          f"C3 sanity check: committed share C1 - C3, {c3['n']} game numbers, mean {f(c3['mean'], False)}, "
          f"two-sided p {fp(c3['p'])} (not a claim).", ""]

    # C1b first breach.
    b = res["c1b"]
    nc = lambda kn: f(kn[0] / kn[1]) if kn[1] else "n/a"
    sel_ba = lambda p, o: f(p[0] / p[1] - o[0] / o[1], False) if p[1] and o[1] else "n/a"
    L += ["## C1b: first breach of the claimed protection (descriptive)", "",
          f"C1b games {b['games']}; proposer dead at the offer {b['dead_proposer']}; games with a first breach "
          f"{b['with_breach']} (rounds after the offer: {sorted(b['breach_rounds'])}; the challenge succeeded in "
          f"{b['breach_succeeded']}). The breach shows X that the claimed protection failed, not why (the proposer "
          "lied or the other player broke the arrangement); it is not called an exposure of the lie.", "",
          "| X's decisions (pooled over C1b games) | non-challenge on proposer plays | non-challenge on other LLM plays "
          "| selectivity |", "|---|---|---|---|",
          f"| before the first breach | {nc(b['before'])} ({b['before'][1]}) | {nc(b['before_others'])} "
          f"({b['before_others'][1]}) | {sel_ba(b['before'], b['before_others'])} |",
          f"| from the first breach on | {nc(b['after'])} ({b['after'][1]}) | {nc(b['after_others'])} "
          f"({b['after_others'][1]}) | {sel_ba(b['after'], b['after_others'])} |", ""]

    # Keyword flags.
    mt = res["mentions"]
    L += ["## Reply wording flags (descriptive keyword counts, not judgements)", "",
          f"Non-LLM / scripted / host reference (RQ 3.3 link), regex: `{NONLLM_RE.pattern}`. C1b only: mention of "
          f"the other players (regex `{OTHERS_RE.pattern}` or another agent's name) and confirmation requests "
          f"(regex `{CONFIRM_RE.pattern}`). All case-insensitive.", "",
          "| cond | replies | replies with non-LLM reference | challenge reasons on proposer plays | reasons with "
          "non-LLM reference | C1b: mention other players | C1b: confirmation request |",
          "|---|---|---|---|---|---|---|"]
    for c in [x for x in conds if x in MESSAGE_CONDS]:
        m = mt[c]
        extra = (f"{m['others']} | {m['confirm']}") if c == "C1b" else "– | –"
        L.append(f"| {c} | {m['replies']} | {m['nonllm']} | {m['reasons']} | {m['reasons_nonllm']} | {extra} |")
    L.append("")

    # Bluff split.
    L += ["## Non-challenge on proposer plays by proposer bluff (used decisions of proposer-facing agents)", "",
          "| cond | bluff plays: decisions, non-challenge | honest plays: decisions, non-challenge |", "|---|---|---|"]
    for c in conds:
        sel_rows = [r for r in rows if r["condition"] == c and r["used"] and r["challenged_is_proposer"]]
        bl = [r for r in sel_rows if r["proposer_bluffed"]]
        ho = [r for r in sel_rows if r["proposer_bluffed"] is False]
        ncr = lambda rs: sum(not r["was_challenged"] for r in rs) / len(rs) if rs else None
        L.append(f"| {c} | {len(bl)}, {f(ncr(bl))} | {len(ho)}, {f(ncr(ho))} |")
    L.append("")

    L += ["## Fallback rates (LLM agents, all decisions)", "", "| agent | challenge decisions | fallbacks |", "|---|---|---|"]
    for ag in sorted({r["challenger"] for r in rows if r["valid"]}):
        dd = [r for r in rows if r["valid"] and r["challenger"] == ag]
        n = sum(r["is_fallback"] for r in dd)
        L.append(f"| {ag} | {len(dd)} | {n} ({f(n / len(dd))}) |")
    L.append("")

    plays = [(c, p) for (c, _), rec in sorted(data.games.items()) for p in rec["scripted_offer"]["plays"]
             if not p["is_scripted"]]
    L += ["## Play fallbacks (LLM agents, all plays)", "", "Cells: fallback plays / LLM plays.", "",
          "| agent | " + " | ".join(conds) + " | total |", "|---|" + "---|" * (len(conds) + 1)]
    for ag in sorted({p["player"] for _, p in plays}):
        cells, tk, tn = [], 0, 0
        for c in conds:
            ps = [p for cc, p in plays if cc == c and p["player"] == ag]
            k = sum(bool(p["is_fallback"]) for p in ps)
            cells.append(f"{k}/{len(ps)}")
            tk, tn = tk + k, tn + len(ps)
        L.append(f"| {ag} | " + " | ".join(cells) + f" | {tk}/{tn} |")
    cells = [f"{sum(bool(p['is_fallback']) for cc, p in plays if cc == c)}/{sum(cc == c for cc, _ in plays)}"
             for c in conds]
    L.append("| all | " + " | ".join(cells) + f" | {sum(bool(p['is_fallback']) for _, p in plays)}/{len(plays)} |")
    L.append("")

    att = defaultdict(Counter)
    for r in runs:
        a = att[r.get("condition")]
        a["attempts"] += 1
        a["failed"] += r.get("status") != "ok"
        a["interrupted"] += r.get("error") == "interrupted"
    L += ["## Game attempts and API errors", "",
          "One attempt = one run of one game (run_log.jsonl). API errors never become game actions: an attempt "
          "that hits a persistent or fatal API error ends with status=error and the game is rerun as a new attempt. "
          "API errors and retries are counted over all attempts.", ""]
    # The runner appends to api_errors.jsonl only when an API error happens, so a run without errors has no
    # such file. Its per-attempt counts are then taken from run_log.jsonl (api_errors, api_retries, api_error);
    # a condition with any run_log row lacking those fields (older runners) shows "–".
    if api_events is None:
        L += ["api_errors.jsonl not present (written only when an API error occurs): API error counts below are "
              "taken from the per-attempt fields of run_log.jsonl; \"–\" = not logged for that condition.", ""]
    L += ["| cond | attempts | failed attempts | interrupted | API errors | API retries | attempts that gave up on the API |",
          "|---|---|---|---|---|---|---|"]
    for c in conds:
        a = att[c]
        if api_events is None:
            rr = [r for r in runs if r.get("condition") == c]
            if rr and all("api_errors" in r and "api_retries" in r for r in rr):
                api = (sum(r["api_errors"] for r in rr), sum(r["api_retries"] for r in rr),
                       sum(r.get("api_error") is not None for r in rr))
            else:
                api = ("–", "–", "–")
        else:
            ev = [e for e in api_events if e.get("condition") == c]
            gave_up = len({e["attempt_id"] for e in ev if e.get("action") == "raise"})
            api = (len(ev), sum(e.get("action") == "retry" for e in ev), gave_up)
        L.append(f"| {c} | {a['attempts']} | {a['failed']} | {a['interrupted']} | {api[0]} | {api[1]} | {api[2]} |")
    L.append("")

    by_model = defaultdict(Counter)
    for u in usage:
        m = by_model[u["model"]]
        m["calls"] += 1
        m["prompt_tokens"] += u.get("prompt_tokens") or 0
        m["completion_tokens"] += u.get("completion_tokens") or 0
        m["latency_s"] += u.get("latency_s") or 0
        m["errors"] += u["status"] != "ok"
    ok_secs = [r["seconds"] for r in runs if r["status"] == "ok" and r.get("seconds") is not None]
    L += ["## API usage", "", "| model | calls | prompt tok | completion tok | errors | mean latency s |",
          "|---|---|---|---|---|---|"]
    for k, v in sorted(by_model.items()):
        L.append(f"| {k} | {v['calls']} | {v['prompt_tokens']} | {v['completion_tokens']} | {v['errors']} | "
                 f"{v['latency_s'] / v['calls']:.2f} |")
    if ok_secs:
        L += ["", f"Wall-clock per game: mean {sum(ok_secs) / len(ok_secs):.0f}s; "
                  f"{len(usage) / len(ok_secs):.0f} LLM calls per completed game."]
    return "\n".join(L) + "\n", by_model, ok_secs


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_judgements(path):
    """judgements.csv -> {(condition, game_number, agent, round): {"category", "reply_sha256"}}.

    When a reply was judged more than once (which the main run must not do), the last row wins and
    a warning is printed.
    """
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["condition"], int(r["game_number"]), r["agent"], int(r["round"]))
            if key in out:
                print(f"WARNING: {key} judged more than once in {path}; using the last row")
            out[key] = {"category": r["category"], "reply_sha256": r.get("reply_sha256")}
    return out


def load_games(records: Path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(records.glob("C*_g??.json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judgements", default=None, help="judgements.csv from judge_replies.py (optional)")
    ap.add_argument("--judge-validated", action="store_true",
                    help="the judge passed its pre-registered validation; without this flag HA1a/b/c are not tested")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = Path(args.records)
    games = load_games(records)
    judgements = load_judgements(args.judgements) if args.judgements else None
    res = compute(games, judgements, args.judge_validated)

    def jsonl(p):
        rows = []
        for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:   # a line cut off by a crash
                    pass
        return rows

    def dedup(rows):
        """Drop repeated event rows (same stream, attempt_id, event_seq); rows without ids are kept."""
        seen, outp = set(), []
        for r in rows:
            key = (r.get("stream"), r.get("attempt_id"), r.get("event_seq"))
            if r.get("attempt_id") is not None and key in seen:
                continue
            seen.add(key)
            outp.append(r)
        return outp

    usage = dedup(jsonl(records / "usage.jsonl"))
    api_path = records / "api_errors.jsonl"
    api_events = dedup(jsonl(api_path)) if api_path.exists() else None
    runs = [r for r in jsonl(records / "run_log.jsonl") if not r.get("mock")]
    data = res["data"]
    offers = []
    for rec in games:
        so = rec["scripted_offer"]
        for o in so["offers"]:
            offers.append({"condition": so["condition"], "game_number": so["game_number"], **o,
                           "judged_category": data.category(so["condition"], so["game_number"], o["agent"])})
    write_csv(out / "offers.csv", offers, ["condition", "game_number", "agent", "model", "round", "decision",
                                           "reply", "parse_error", "judged_category", "raw_response"])
    write_csv(out / "challenges.csv", data.rows,
              ["condition", "game_number", "seed", "round", "challenger", "challenged", "challenged_is_proposer",
               "acceptance_status", "challenger_can_challenge_proposer", "proposer_bluffed", "was_challenged",
               "is_fallback", "is_system", "challenger_is_scripted", "post_offer", "valid", "used",
               "exclusion_reason", "challenge_reason", "after_first_breach"])
    write_csv(out / "selectivity.csv", res["sel_rows"],
              ["condition", "game_number", "agents", "nc_proposer", "n_proposer", "nc_others", "n_others", "selectivity"])
    write_csv(out / "paired_values.csv", res["paired_rows"],
              ["hypothesis", "condition", "game_number", "value", "n_treated", "n_control"])
    text, by_model, ok_secs = render(res, usage, runs, api_events)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "usage_summary.json").write_text(json.dumps({
        "by_model": {k: dict(v) for k, v in by_model.items()}, "total_calls": len(usage),
        "seconds_per_game": ok_secs, "run_errors": [r for r in runs if r["status"] != "ok"]}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
