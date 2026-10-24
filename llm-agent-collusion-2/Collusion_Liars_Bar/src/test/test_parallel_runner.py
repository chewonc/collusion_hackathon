"""Offline checks for run_scripted_offer.py --workers K (no network; uses MockLLMClient).

The mock LLM's reply is a hash of the prompt, so a game's record is a deterministic function of
its seed, condition and code path. Sequential and parallel runs must therefore give identical
records (apart from game_id, a wall-clock timestamp).

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_parallel_runner.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.game.run_scripted_offer import execution_order, _merge_record  # noqa: E402

CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
GAMES = 6
SIDE_DIR = ROOT / "experiments" / "game_records"


def run(records, results, *extra):
    cmd = [sys.executable, "src/game/run_scripted_offer.py", "--mock", "--games", str(GAMES),
           "--records-dir", str(records), "--results-dir", str(results), *extra]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out


def records(d):
    recs = {}
    for p in sorted(Path(d).glob("C?_g??.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec.pop("game_id")   # wall-clock timestamp: the only timing field in a record
        recs[p.name] = rec
    return recs


def run_log(d):
    return [json.loads(l) for l in (Path(d) / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def side_files():
    return {p.name for p in SIDE_DIR.glob("*.json")} if SIDE_DIR.exists() else set()


def main():
    expected_jobs = [(c, g) for g in range(GAMES) for c in execution_order(CONDITIONS, g)]
    expected_names = {f"{c}_g{g:02d}.json" for c, g in expected_jobs}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        res = tmp / "results"

        # 1. Sequential reference, then two parallel runs with different pool sizes.
        run(tmp / "seq", res)
        before = side_files()
        run(tmp / "par10", res, "--workers", "10")
        run(tmp / "par3", res, "--workers", "3")
        assert side_files() == before, "parallel games must not write upstream auto-save files into the shared dir"
        seq, par10, par3 = records(tmp / "seq"), records(tmp / "par10"), records(tmp / "par3")

        # 2. No missing, extra or duplicate records; nothing left in the worker dirs.
        for name, recs in (("seq", seq), ("par10", par10), ("par3", par3)):
            assert set(recs) == expected_names, f"{name}: records {sorted(set(recs) ^ expected_names)}"
        for d in ("par10", "par3"):
            leftovers = list((tmp / d / "_workers").rglob("C?_g??.json"))
            assert not leftovers, f"unmerged records: {leftovers}"
        print(f"ok {len(expected_names)} records in each run; no missing, extra or unmerged records")

        # 3. Records identical to the sequential run (excluding game_id).
        for name in sorted(expected_names):
            assert par10[name] == seq[name], f"{name}: parallel (10 workers) record differs from sequential"
            assert par3[name] == seq[name], f"{name}: parallel (3 workers) record differs from sequential"
        print("ok every record identical to the sequential run with 10 and with 3 workers (game_id excluded)")

        # 4. run_log: one line per job, sequential job order and counterbalancing, one fresh process per game.
        seq_log = run_log(tmp / "seq")
        for d in ("par10", "par3"):
            log = run_log(tmp / d)
            assert [(r["condition"], r["game_number"]) for r in log] == expected_jobs, f"{d}: job order"
            for a, b in zip(log, seq_log):
                for k in ("condition", "game_number", "seed", "execution_position", "execution_order",
                          "status", "rounds", "winner"):
                    assert a[k] == b[k], f"{d}: run_log field {k} differs for {a['condition']} g{a['game_number']}"
            assert all(r["execution_order"] == execution_order(CONDITIONS, r["game_number"]) for r in log)
            assert all(r["execution_order"][r["execution_position"]] == r["condition"] for r in log)
            pids = Counter(r["pid"] for r in log)
            assert len(pids) == len(log), f"{d}: a process ran more than one game: {pids.most_common(3)}"
        balance = Counter((r["condition"], r["execution_position"]) for r in run_log(tmp / "par10"))
        assert all(balance[(c, p)] >= 1 for c in CONDITIONS for p in range(5)), "each condition in each position"
        print("ok run_log: one line per game, sequential job order and counterbalanced positions, "
              "a distinct process for every game")

        # 5. Pairing and RNG: matched games share seating and round-1 state; game numbers differ.
        r1 = {}
        for name, rec in par10.items():
            r = rec["rounds"][0]
            r1[name] = (r["target_card"], r["starting_player"],
                        [(s["player_name"], s["bullet_position"], s["initial_hand"]) for s in r["player_initial_states"]],
                        rec["scripted_offer"]["seat_order"])
        for g in range(GAMES):
            states = {repr(r1[f"{c}_g{g:02d}.json"]) for c in CONDITIONS}
            assert len(states) == 1, f"g{g}: matched games differ in seating or round-1 state"
        assert len({repr(r1[f"C4_g{g:02d}.json"]) for g in range(GAMES)}) == GAMES, \
            "different game numbers should have different round-1 states"
        print("ok within each game number all conditions share seating and round-1 deal/bullets/target; "
              "game numbers differ")

        # 6. Logs: each game's log holds only that game.
        for c, g in expected_jobs:
            text = (tmp / "par10" / "logs" / f"{c}_g{g:02d}.log").read_text(encoding="utf-8")
            heads = re.findall(r"Scripted-offer game: condition=(C\d) game=(\d+) seed=(\d+)", text)
            assert heads == [(c, str(g), str(g))], f"{c} g{g}: log headers {heads}"
            offers = set(re.findall(r"\[SCRIPTED OFFER\] (C\d) ", text))
            assert offers <= {c}, f"{c} g{g}: log mentions offers of {offers}"
        print("ok every log contains exactly its own game (one header, no other condition's offers)")

        # 7. Resume: delete some records and rerun; only those are re-created, identically.
        for name in ("C2_g01.json", "C0_g04.json"):
            (tmp / "par10" / name).unlink()
        n_before = len(run_log(tmp / "par10"))
        out = run(tmp / "par10", res, "--workers", "10")
        assert out.stdout.count("skip ") == len(expected_names) - 2, "resume must skip existing records"
        assert len(run_log(tmp / "par10")) == n_before + 2
        assert records(tmp / "par10") == seq, "resumed records must match the sequential run"
        print("ok resume re-creates only the missing records, identically")

        # 8. Merge guard: missing, mismatched and duplicate records are rejected, never overwritten.
        d = tmp / "guard"
        work = d / "_workers" / "C1_g03"
        work.mkdir(parents=True)
        assert "missing" in _merge_record(d, "C1", 3)
        good = (tmp / "seq" / "C1_g03.json").read_text(encoding="utf-8")
        (work / "C1_g03.json").write_text(good.replace('"game_number": 3', '"game_number": 4'), encoding="utf-8")
        assert "mismatch" in _merge_record(d, "C1", 3)
        (work / "C1_g03.json").write_text(good, encoding="utf-8")
        (d / "C1_g03.json").write_text("existing", encoding="utf-8")
        assert "duplicate" in _merge_record(d, "C1", 3)
        assert (d / "C1_g03.json").read_text(encoding="utf-8") == "existing", "an existing record was overwritten"
        (d / "C1_g03.json").unlink()
        assert _merge_record(d, "C1", 3) is None and json.loads((d / "C1_g03.json").read_text(encoding="utf-8"))
        print("ok merge rejects missing, mismatched and duplicate records without overwriting")
    print("ALL PARALLEL CHECKS PASSED")


if __name__ == "__main__":
    main()
