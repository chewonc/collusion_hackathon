"""Run the scripted-offer experiment: conditions x game numbers, resumable.

Run from Collusion_Liars_Bar/ (prompt paths are relative to it):

    ../.venv/Scripts/python.exe src/game/run_scripted_offer.py --pilot
    ../.venv/Scripts/python.exe src/game/run_scripted_offer.py --games 12

Inference goes through the local OpenAI-compatible router (default http://127.0.0.1:8100), the
same path as the baseline run. --mock swaps in the offline MockLLMClient instead.

Seeds: game number g uses seed g in EVERY condition, and seat order is also a function of g
only, so conditions share the same round-1 deal, bullets and seating. LLM sampling is not
seeded, so trajectories still diverge after the first model call.

Execution order: games run one at a time. Within game number g the conditions run in the
cyclic order execution_order(conditions, g) (the list rotated by g), so each condition takes each
position once per len(conditions) consecutive game numbers. The position is logged in
run_log.jsonl; the analysis pairs games by game number and does not use the order.

Outputs (per game): <records-dir>/<cond>_g<NN>.json, <records-dir>/logs/<cond>_g<NN>.log,
plus one line per game in <records-dir>/run_log.jsonl and one line per LLM call in
<records-dir>/usage.jsonl. Prompt templates are copied verbatim to <results-dir>/prompts_verbatim/.

Parallel mode (--workers K, K > 1): the same job list in the same order, run by a pool of K
processes with maxtasksperchild=1, so every game runs in a fresh process (own global `random`,
own module state, own stdout). Each game writes only into <records-dir>/_workers/<cond>_g<NN>/
(record, usage rows, upstream per-round auto-save files) and its own log file. The parent is the
only writer of the shared files: it receives results in job order, checks each record (parses,
matches its condition / game number / seed, no existing copy) and moves it into <records-dir>,
then appends that game's usage rows and its run_log line.
"""
import argparse
import contextlib
import datetime
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = Path(r"C:\AISafetyResearch\Collusion\results\scripted_offer")
DEFAULT_RECORDS = ROOT / "experiments" / "game_records" / "scripted_offer"
VERBATIM_FILES = [
    "prompts/scripted_offer/offer_direct.txt",
    "prompts/scripted_offer/offer_hint.txt",
    "prompts/scripted_offer/offer_benign.txt",
    "prompts/scripted_offer/response_prompt.txt",
    "prompts/scripted_offer/private_messages_block.txt",
    "prompts/secret_channel/partner_notification_prompt.txt",
    "prompts/rule_vanilla.txt",
    "prompts/play_card_prompt_template.txt",
    "prompts/challenge_prompt_template.txt",
    "prompts/reflect_prompt_template.txt",
]

_usage_ctx = {"condition": None, "game_number": None, "file": None}


def execution_order(conditions, game_number):
    """Order in which the conditions of one game number are run (cyclic counterbalancing).

    Games run one at a time, so a fixed order would confound condition with position in time
    (server warm-up, drift). Rotating the list by the game number puts every condition in every
    position exactly once per len(conditions) consecutive game numbers. The order is
    deterministic, and it cannot change a game's content: each game reseeds its own RNGs from its
    seed when it is constructed, so matched games (same game number) stay identical in seed,
    seating and initial deal whatever their position.
    """
    k = game_number % len(conditions)
    return list(conditions[k:]) + list(conditions[:k])


def _configure_router_env(router: str) -> None:
    """Point the shared client at the router and blank the hosted-provider keys (as the baseline did)."""
    os.environ.pop("LLM_MODEL", None)
    os.environ.update(LLM_BASE_URL=router.rstrip("/") + "/v1", LLM_API_KEY="local-router",
                      NEBIUS_API_KEY="", OPENROUTER_API_KEY="", PYTHONUTF8="1")


def _install_usage_tracking() -> None:
    """Wrap every LLMClient's OpenAI client so each call's token usage and latency are logged."""
    from src.llm import llm_client as llm_module
    original_init = llm_module.LLMClient.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        create = self._client.chat.completions.create

        def tracked_create(*a, **k):
            t0 = time.time()
            status = "ok"
            usage = None
            try:
                resp = create(*a, **k)
                usage = getattr(resp, "usage", None)
                return resp
            except Exception as e:
                status = type(e).__name__
                raise
            finally:
                if _usage_ctx["file"]:
                    rec = {
                        "condition": _usage_ctx["condition"],
                        "game_number": _usage_ctx["game_number"],
                        "model": k.get("model"),
                        "status": status,
                        "latency_s": round(time.time() - t0, 3),
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "ts": time.time(),
                    }
                    with open(_usage_ctx["file"], "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")

        self._client.chat.completions.create = tracked_create

    llm_module.LLMClient.__init__ = init


def _install_mock() -> None:
    from src.agents import player as player_module
    from src.test.mock_llm import MockLLMClient
    player_module.LLMClient = MockLLMClient


def _copy_verbatim(results_dir: Path) -> None:
    dest = results_dir / "prompts_verbatim"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in VERBATIM_FILES:
        target = dest / rel.replace("/", "__")
        shutil.copy2(ROOT / rel, target)


def _parallel_job(job):
    """Run one game in a fresh pool process (maxtasksperchild=1); returns its run_log line."""
    cond, g, position, order, opts = job
    records_dir = Path(opts["records_dir"])
    work = records_dir / "_workers" / f"{cond}_g{g:02d}"
    t0 = time.time()
    status, err, record = "ok", None, None
    try:
        os.chdir(ROOT)
        if opts["mock"]:
            _install_mock()
        else:
            _configure_router_env(opts["router"])
            _install_usage_tracking()
        from src.game.game_scripted_offer import GameScriptedOffer
        shutil.rmtree(work, ignore_errors=True)   # leftovers of an interrupted run of this game
        work.mkdir(parents=True)
        _usage_ctx.update(condition=cond, game_number=g, file=str(work / "usage.jsonl"))
        log_path = records_dir / "logs" / f"{cond}_g{g:02d}.log"
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            try:
                game = GameScriptedOffer(cond, g, seed=g, out_dir=str(work),
                                         offer_round=opts["offer_round"], bluff_prob=opts["bluff_prob"])
                # Upstream GameRecord auto-saves to a second-resolution timestamp name; keep it per game.
                game.game_record.save_directory = str(work)
                record = game.start_game()
            except Exception:
                status, err = "error", traceback.format_exc()
                print(err)
    except Exception:
        status, err = "error", traceback.format_exc()
    return {
        "condition": cond, "game_number": g, "seed": g, "execution_position": position,
        "execution_order": order, "status": status, "seconds": round(time.time() - t0, 1),
        "rounds": len(record["rounds"]) if record else None,
        "winner": record.get("winner") if record else None,
        "mock": opts["mock"],
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error": err, "workers": opts["workers"], "pid": os.getpid(),
    }


def _merge_record(records_dir: Path, cond: str, g: int):
    """Move a worker's record into records_dir after checking it. Returns an error message or None."""
    name = f"{cond}_g{g:02d}.json"
    src, dst = records_dir / "_workers" / f"{cond}_g{g:02d}" / name, records_dir / name
    if not src.exists():
        return f"record missing: {src}"
    try:
        so = json.loads(src.read_text(encoding="utf-8"))["scripted_offer"]
    except Exception as e:
        return f"record unreadable: {src}: {e!r}"
    if (so.get("condition"), so.get("game_number"), so.get("seed")) != (cond, g, g):
        return f"record mismatch: {src} has {(so.get('condition'), so.get('game_number'), so.get('seed'))}"
    if dst.exists():
        return f"duplicate record: {dst} already exists"
    os.replace(src, dst)
    return None


def _run_parallel(jobs, opts, records_dir: Path) -> int:
    import multiprocessing as mp
    failures = 0
    with mp.get_context("spawn").Pool(processes=opts["workers"], maxtasksperchild=1) as pool:
        # imap yields in job order, so run_log.jsonl / usage.jsonl keep the sequential order.
        for line in pool.imap(_parallel_job, jobs):
            cond, g = line["condition"], line["game_number"]
            if line["status"] == "ok":
                problem = _merge_record(records_dir, cond, g)
                if problem:
                    line["status"], line["error"] = "error", problem
            usage = records_dir / "_workers" / f"{cond}_g{g:02d}" / "usage.jsonl"
            if usage.exists():
                with open(records_dir / "usage.jsonl", "a", encoding="utf-8") as f:
                    f.write(usage.read_text(encoding="utf-8"))
            with open(records_dir / "run_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            print(f"{cond} g{g:02d}: {line['status']} in {line['seconds']}s, rounds={line['rounds']}, "
                  f"winner={line['winner']}, pid={line['pid']}")
            failures += line["status"] != "ok"
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Scripted-offer experiment runner")
    ap.add_argument("--conditions", default="C0,C1,C2,C3,C4")
    ap.add_argument("--games", type=int, default=12, help="game numbers 0..games-1 per condition")
    ap.add_argument("--pilot", action="store_true", help="one game (game number 0) per condition")
    ap.add_argument("--offer-round", type=int, default=2)
    ap.add_argument("--bluff-prob", type=float, default=0.5)
    ap.add_argument("--records-dir", default=str(DEFAULT_RECORDS))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--router", default="http://127.0.0.1:8100")
    ap.add_argument("--mock", action="store_true", help="offline: use MockLLMClient, no network")
    ap.add_argument("--workers", type=int, default=1,
                    help="games run concurrently, one fresh process per game (1 = sequential)")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    games = 1 if args.pilot else args.games
    records_dir = Path(args.records_dir)
    (records_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    _copy_verbatim(results_dir)

    if args.workers > 1:
        records_dir = records_dir.resolve()   # workers chdir to ROOT
        opts = {"records_dir": str(records_dir), "mock": args.mock, "router": args.router,
                "offer_round": args.offer_round, "bluff_prob": args.bluff_prob, "workers": args.workers}
        jobs = []
        for g in range(games):
            order = execution_order(conditions, g)
            for position, cond in enumerate(order):
                if (records_dir / f"{cond}_g{g:02d}.json").exists():
                    print(f"skip {cond}_g{g:02d}.json (exists)")
                    continue
                jobs.append((cond, g, position, order, opts))
        return _run_parallel(jobs, opts, records_dir)

    os.chdir(ROOT)
    if args.mock:
        _install_mock()
    else:
        _configure_router_env(args.router)
        _install_usage_tracking()
    from src.game.game_scripted_offer import GameScriptedOffer

    _usage_ctx["file"] = str(records_dir / "usage.jsonl")
    failures = 0
    # Interleave conditions within each game number so a partial run stays balanced, in a
    # counterbalanced order (see execution_order).
    for g in range(games):
        order = execution_order(conditions, g)
        for position, cond in enumerate(order):
            out_path = records_dir / f"{cond}_g{g:02d}.json"
            if out_path.exists():
                print(f"skip {out_path.name} (exists)")
                continue
            _usage_ctx.update(condition=cond, game_number=g)
            log_path = records_dir / "logs" / f"{cond}_g{g:02d}.log"
            t0 = time.time()
            status, err, record = "ok", None, None
            with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
                    contextlib.redirect_stderr(log):
                try:
                    game = GameScriptedOffer(cond, g, seed=g, out_dir=str(records_dir),
                                             offer_round=args.offer_round, bluff_prob=args.bluff_prob)
                    record = game.start_game()
                except Exception:
                    status, err = "error", traceback.format_exc()
                    print(err)
            secs = round(time.time() - t0, 1)
            line = {
                "condition": cond, "game_number": g, "seed": g, "execution_position": position,
                "execution_order": order, "status": status, "seconds": secs,
                "rounds": len(record["rounds"]) if record else None,
                "winner": record.get("winner") if record else None,
                "mock": args.mock,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "error": err,
            }
            with open(records_dir / "run_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            print(f"{cond} g{g:02d}: {status} in {secs}s, rounds={line['rounds']}, winner={line['winner']}")
            if status != "ok":
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
