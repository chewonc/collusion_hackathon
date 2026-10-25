"""Offline checks of the counterbalanced execution order (no network; MockLLMClient).

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_run_order.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.agents import player as player_module
from src.test.mock_llm import MockLLMClient

player_module.LLMClient = MockLLMClient

from src.game.game_scripted_offer import CONDITIONS, GameScriptedOffer
from src.game.run_scripted_offer import execution_order


def test_balance():
    for n_games in (5, 10, 12):
        pos = Counter()
        for g in range(n_games):
            order = execution_order(list(CONDITIONS), g)
            assert sorted(order) == sorted(CONDITIONS), "each order must be a permutation"
            for i, c in enumerate(order):
                pos[(c, i)] += 1
        counts = [pos[(c, i)] for c in CONDITIONS for i in range(len(CONDITIONS))]
        assert max(counts) - min(counts) <= 1, (n_games, counts)
        if n_games % len(CONDITIONS) == 0:
            assert set(counts) == {n_games // len(CONDITIONS)}
    assert execution_order(list(CONDITIONS), 7) == execution_order(list(CONDITIONS), 7)
    assert execution_order(["C4"], 3) == ["C4"]
    print("ok order: permutation per game number, every condition in every position equally often "
          "(within 1), deterministic")


def run_direct(order, g, out_dir):
    recs = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for c in order:
            rec = GameScriptedOffer(c, g, seed=g, out_dir=out_dir).start_game()
            rec.pop("game_id", None)
            recs[c] = rec
    return recs


def test_order_invariance(tmp):
    for g in (1, 2):
        a = run_direct(list(CONDITIONS), g, tmp)
        b = run_direct(list(reversed(CONDITIONS)), g, tmp)
        for c in CONDITIONS:
            assert a[c] == b[c], f"record for {c} g{g} depends on execution order"
        r1 = {c: (a[c]["rounds"][0]["target_card"], a[c]["rounds"][0]["starting_player"],
                  [(s["player_name"], s["bullet_position"], s["initial_hand"]) for s in a[c]["rounds"][0]["player_initial_states"]],
                  a[c]["scripted_offer"]["seat_order"]) for c in CONDITIONS}
        assert len(set(map(repr, r1.values()))) == 1, "matched games must share seed, seating and initial deal"
    print("ok matched games identical whatever the execution order; same seed, seating and round-1 deal")


def test_runner_uses_order(tmp):
    rec_dir, res_dir = os.path.join(tmp, "records"), os.path.join(tmp, "results")
    cmd = [sys.executable, "src/game/run_scripted_offer.py", "--mock", "--games", "3",
           "--records-dir", rec_dir, "--results-dir", res_dir]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    log = [json.loads(l) for l in open(os.path.join(rec_dir, "run_log.jsonl"), encoding="utf-8")]
    expected = [(c, g) for g in range(3) for c in execution_order(list(CONDITIONS), g)]
    assert [(r["condition"], r["game_number"]) for r in log] == expected, "runner must follow execution_order"
    assert all(r["execution_order"] == execution_order(list(CONDITIONS), r["game_number"]) for r in log)
    assert all(r["execution_order"][r["execution_position"]] == r["condition"] for r in log)
    direct = run_direct(list(CONDITIONS), 2, os.path.join(tmp, "direct"))
    for c in CONDITIONS:
        rec = json.load(open(os.path.join(rec_dir, f"{c}_g02.json"), encoding="utf-8"))
        rec.pop("game_id", None)
        # attempt_id is record-only run metadata (None in a direct run).
        assert rec["scripted_offer"].pop("attempt_id") == f"{c}_g02_a1"
        assert direct[c]["scripted_offer"].pop("attempt_id", None) is None
        assert rec == direct[c], f"runner record {c} g2 differs from a direct run"
    print("ok runner executes and logs the counterbalanced order; its records equal direct runs")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_balance()
        test_order_invariance(tmp)
        test_runner_uses_order(tmp)
    print("ALL ORDER CHECKS PASSED")
