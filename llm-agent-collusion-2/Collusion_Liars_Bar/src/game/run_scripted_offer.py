"""Run the scripted-offer experiment: conditions x game numbers, resumable.

Run from Collusion_Liars_Bar/ (prompt paths are relative to it):

    ../.venv/Scripts/python.exe src/game/run_scripted_offer.py --pilot
    ../.venv/Scripts/python.exe src/game/run_scripted_offer.py --games 12

Inference goes through the local OpenAI-compatible router (default http://127.0.0.1:8100), the
same path as the baseline run. --mock swaps in the offline MockLLMClient instead.

Seeds: game number g uses seed g in EVERY condition, and seat order is also a function of g
only, so conditions share the same round-1 deal, bullets and seating. LLM sampling is not
seeded, so trajectories still diverge after the first model call.

Execution order: within game number g the conditions are queued in the cyclic order
execution_order(conditions, g) (the list rotated by g), so each condition takes each position
once per len(conditions) consecutive game numbers. The position is logged in run_log.jsonl; the
analysis pairs games by game number and does not use the order.

Attempts. Every run of one game is an attempt with its own id, <cond>_g<NN>_a<n>:
  * n is 1 + the largest attempt number already used for that (condition, game number), read
    from run_log.jsonl, usage.jsonl, api_errors.jsonl and the attempt directories; other games'
    rows are ignored. Only the runner holding the runner lock assigns numbers.
  * An attempt is reserved before it starts by creating <records-dir>/_workers/<attempt_id>/
    (os.mkdir, fails if it exists) and attempt.json inside it (O_EXCL). A failed reservation is
    logged as an error and never deletes or overwrites anything.
  * The game writes only into its attempt directory (record, usage rows, api_errors rows,
    upstream per-round auto-save files) and its log file; a previous log of the same game is
    renamed to <cond>_g<NN>.attempt<n>.log first, so no log is overwritten.
  * When an attempt ends, successful or not, the parent, in this order: moves a checked record
    into <records-dir>; appends and fsyncs the run_log line; merges the attempt's usage and
    api_errors rows into <records-dir> idempotently (key: stream, attempt_id, event_seq) and checks
    them; only then moves the attempt directory to <records-dir>/_attempts/ (nothing is deleted).
  * An API error never becomes a game action (see experiment_player.StrictLLMClient): the game
    stops, the attempt is logged with status=error, and rerunning the runner reruns that game.

One runner per records directory: the runner takes an OS file lock on <records-dir>/.runner.lock
(released by the OS when the process ends, however it ends) and refuses to start if another
runner holds it (exit code 3). Each game holds an OS lock on <attempt dir>/worker.lock while it
runs. At start, the runner recovers attempt directories left by a runner that died: one whose
worker lock can be taken is merged and logged as "interrupted" (or "ok" if its record had already
been moved into <records-dir>); one whose worker is still alive is left alone and that game is
not queued. Lock semantics need a local disk: do not put <records-dir> on a network drive.

Parallel mode (--workers K, K > 1): the same job queue, run by a pool of K processes with
maxtasksperchild=1, so every game runs in a fresh process (own global `random`, own module
state, own stdout). Results are taken in completion order, so a game stuck in API backoff does
not hold up the merging of games that finished. The parent is the only writer of the shared
files. With --workers 1 the same steps run in this process, one game at a time; that serial mode
is accepted only with --mock (offline tests; their default), so every real run is parallel
(default --workers 10).

Outputs (per game): <records-dir>/<cond>_g<NN>.json, <records-dir>/logs/<cond>_g<NN>.log, one
line per attempt in <records-dir>/run_log.jsonl, one line per LLM call in usage.jsonl and one
line per API error in api_errors.jsonl. Prompt templates are copied verbatim to
<results-dir>/prompts_verbatim/.
"""
import argparse
import contextlib
import datetime
import json
import os
import queue
import re
import socket
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = Path(r"X:\Collusion\results\scripted_offer")
DEFAULT_RECORDS = ROOT / "experiments" / "game_records" / "scripted_offer"
VERBATIM_FILES = [
    "prompts/scripted_offer/offer_direct.txt",
    "prompts/scripted_offer/offer_hint.txt",
    "prompts/scripted_offer/offer_benign.txt",
    "prompts/scripted_offer/offer_broker.txt",
    "prompts/scripted_offer/offer_placebo.txt",
    # Family B (Amendment 14)
    "prompts/scripted_offer/offer_consistency.txt",
    "prompts/scripted_offer/offer_consistency_instruct.txt",
    "prompts/scripted_offer/offer_authority.txt",
    "prompts/scripted_offer/series_memory_block.txt",
    "prompts/scripted_offer/response_prompt.txt",
    "prompts/scripted_offer/private_messages_block.txt",
    "prompts/secret_channel/partner_notification_prompt.txt",
    "prompts/rule_vanilla.txt",
    "prompts/play_card_prompt_template.txt",
    "prompts/challenge_prompt_template.txt",
    "prompts/reflect_prompt_template.txt",
]
EXIT_LOCKED = 3
EXIT_WARMUP = 4
# Wakes every backend at once before game 1 (the serial pilot paid three cold starts inside its first
# game: 549 s vs a 309 s mean). The probes go to the backends directly and never through the router,
# so they appear in no record, usage row or router log; they change nothing a model sees in a game.
DEFAULT_WARMUP_SCRIPT = ROOT.parents[2] / "experiments" / "modal_infra" / "warmup.py"
STREAM_FILES = {"usage": "usage.jsonl", "api_error": "api_errors.jsonl"}
AID_RE = re.compile(r"^(C\d[a-z]*)_g(\d+)_a(\d+)$")   # [a-z]* : C1b, C1c, C1ci, C1d, C1s

# Per-process state of the game being run (one game per process at a time).
_usage_ctx = {"condition": None, "game_number": None, "attempt_id": None, "usage_file": None,
              "events_file": None, "seq": 0}


def execution_order(conditions, game_number):
    """Order in which the conditions of one game number are queued (cyclic counterbalancing).

    A fixed order would confound condition with position in time (server warm-up, drift).
    Rotating the list by the game number puts every condition in every position exactly once per
    len(conditions) consecutive game numbers. The order is deterministic, and it cannot change a
    game's content: each game reseeds its own RNGs from its seed when it is constructed, so
    matched games (same game number) stay identical in seed, seating and initial deal whatever
    their position.
    """
    k = game_number % len(conditions)
    return list(conditions[k:]) + list(conditions[:k])


def attempt_id(cond: str, g: int, n: int) -> str:
    return f"{cond}_g{g:02d}_a{n}"


# ----------------------------------------------------------------------
# HB1 series (PREREG Amendment 14): k consecutive games per game number, true memory from game 2 on
# ----------------------------------------------------------------------
SERIES_SEED_STEP = 1000   # game i of a series uses seed g + (i - 1) * SERIES_SEED_STEP


def series_seed(g: int, i: int) -> int:
    return g + (i - 1) * SERIES_SEED_STEP


def record_name(cond: str, g: int, series: int = 1, i: int = None) -> str:
    """<cond>_gNN.json, or <cond>_gNN_kI.json for game I of a k-game series (I = k when omitted)."""
    if series > 1:
        return f"{cond}_g{g:02d}_k{series if i is None else i}.json"
    return f"{cond}_g{g:02d}.json"


def build_memory_block(records: list, agent: str, k_now: int, template: str) -> str:
    """The memory text an agent carries into game `k_now` of a series: one filled copy of the
    frozen template per earlier game, every number counted from that game's record.

    Lines for the private message and the agent's reply appear only for a game in which the agent
    received a message (game 1); otherwise a single line says that no message was received. Nothing
    is invented: `c` is the number of times the proposer challenged this agent's plays (the proposer
    never challenges, so it is 0), `n_judged_by_proposer` the number of this agent's plays the
    proposer was the judging seat for, `m`/`n` the agent's own non-challenges / decisions on the
    proposer's plays, and the result line names the winner and whether the agent survived.
    """
    blocks = []
    for i, rec in enumerate(records, start=1):
        so = rec["scripted_offer"]
        proposer = so["proposer"]
        offer = next((o for o in so["offers"] if o["agent"] == agent), None)
        dec = so["decisions"]
        judged = [d for d in dec if d["challenger"] == proposer and d["current_player"] == agent]
        own = [d for d in dec if d["challenger"] == agent and d["is_proposer_play"]]
        c = sum(1 for d in judged if d["was_challenged"])
        m = sum(1 for d in own if not d["was_challenged"])
        winner = rec.get("winner")
        last_players = rec["rounds"][-1]["round_players"] if rec.get("rounds") else []
        survived = agent in last_players
        outcome = (f"{winner} won the game; " if winner else "no winner recorded; ") + \
                  ("you were the winner. " if winner == agent else
                   ("you survived to the end. " if survived else "you were eliminated. ")) + \
                  "Final scores: " + ", ".join(f"{k} {v}" for k, v in (rec.get("final_scores") or {}).items()) + "."
        text = template
        if offer is not None:
            reply = offer["reply"] if offer["reply"] else " ".join((offer["raw_response"] or "").split()[:50])
            # The message body only: the delivered text starts with the shared header line
            # "Private message from <proposer> (only you can see it):", which the record line already says.
            body = offer["offer_text"].split("\n", 1)[1].strip() if "\n" in offer["offer_text"] else offer["offer_text"]
            text = (text.replace("{offer_round}", str(offer["round"]))
                        .replace("{offer_text}", body)
                        .replace("{reply_text}", reply or "(no reply)"))
        else:
            lines = [ln for ln in text.split("\n")
                     if "{offer_text}" not in ln and "{reply_text}" not in ln]
            idx = next(j for j, ln in enumerate(lines) if ln.startswith("- "))
            lines.insert(idx, "- No private message from {proposer_name} in this game.")
            text = "\n".join(lines)
        text = (text.replace("{i}", str(i)).replace("{k}", str(k_now))
                    .replace("{proposer_name}", proposer)
                    .replace("{c}", str(c)).replace("{n_judged_by_proposer}", str(len(judged)))
                    .replace("{m}", str(m)).replace("{n}", str(len(own)))
                    .replace("{outcome_line}", outcome))
        blocks.append(text.strip("\n"))
    return "\n\n" + "\n\n".join(blocks) + "\n"


def parse_attempt_id(aid):
    m = AID_RE.match(aid or "")
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def _configure_router_env(router: str) -> None:
    """Point the shared client at the router and blank the hosted-provider keys (as the baseline did)."""
    os.environ.pop("LLM_MODEL", None)
    os.environ.update(LLM_BASE_URL=router.rstrip("/") + "/v1", LLM_API_KEY="local-router",
                      NEBIUS_API_KEY="", OPENROUTER_API_KEY="", PYTHONUTF8="1")


# ----------------------------------------------------------------------
# Event rows (usage and API errors) written by the game process
# ----------------------------------------------------------------------
def _append_line(path, row) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _event_base(stream: str) -> dict:
    """Common fields; event_seq is one counter shared by all streams of an attempt."""
    _usage_ctx["seq"] += 1
    return {"stream": stream, "attempt_id": _usage_ctx["attempt_id"], "event_seq": _usage_ctx["seq"],
            "condition": _usage_ctx["condition"], "game_number": _usage_ctx["game_number"]}


def _api_event_sink(event: dict) -> None:
    if _usage_ctx["events_file"]:
        _append_line(_usage_ctx["events_file"], {**_event_base("api_error"), **event, "ts": time.time()})


def _install_usage_tracking() -> None:
    """Wrap every StrictLLMClient's transport so each call's token usage and latency are logged,
    and send API error events to api_errors.jsonl. Idempotent."""
    from src.agents import experiment_player as ep
    ep.set_api_event_sink(_api_event_sink)
    original_init = ep.StrictLLMClient.__init__
    if getattr(original_init, "_usage_tracked", False):
        return

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
                if _usage_ctx["usage_file"]:
                    _append_line(_usage_ctx["usage_file"], {
                        **_event_base("usage"),
                        "model": k.get("model"),
                        "status": status,
                        "latency_s": round(time.time() - t0, 3),
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "ts": time.time(),
                    })

        self._client.chat.completions.create = tracked_create

    init._usage_tracked = True
    ep.StrictLLMClient.__init__ = init


def _install_mock(fault_spec=None, cond=None, g=None) -> None:
    """Offline mode. Without a matching fault: MockLLMClient, exactly as before. With a fault for
    this game: StrictLLMClient over a fake transport that raises the configured API errors."""
    from src.agents import player as player_module
    from src.agents import experiment_player as ep
    from src.test.mock_llm import MockLLMClient
    player_module.LLMClient = MockLLMClient
    ep.set_strict_client_factory(None)
    if fault_spec:
        from src.test.mock_llm import MockFault, MockTransport
        fault = MockFault(fault_spec)
        if fault.applies(cond, g):
            from src.llm import llm_client as llm_module
            llm_module.resolve_model = lambda m, *a, **k: m   # no config lookup in tests
            marker = os.environ.get("MOCK_HANG_MARKER")
            ep.set_strict_client_factory(lambda: ep.StrictLLMClient(client=MockTransport(fault, marker),
                                                                    sleep=lambda s: None))
            _install_usage_tracking()


def warm_up(script: Path, routes: Path, out_dir: Path) -> bool:
    """Run warmup.py --probe chat as its own process; True if every backend became ready."""
    import subprocess
    if not script.exists():
        print(f"warm-up script not found: {script}", flush=True)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", str(script), "--routes", str(routes), "--out", str(out_dir), "--probe", "chat"]
    print("[warmup] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env={**os.environ, "PYTHONUTF8": "1"}).returncode == 0


def _copy_verbatim(results_dir: Path) -> None:
    import shutil
    dest = results_dir / "prompts_verbatim"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in VERBATIM_FILES:
        target = dest / rel.replace("/", "__")
        shutil.copy2(ROOT / rel, target)


# ----------------------------------------------------------------------
# OS file locks (released by the OS when the holding process ends)
# ----------------------------------------------------------------------
def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _lock_is_free(path: Path) -> bool:
    """True if nobody holds the OS lock on `path` (probe: take it and release it at once)."""
    fh = open(path, "a+b")
    if not _try_lock(fh):
        fh.close()
        return False
    _unlock(fh)
    return True


_RUNNER_LOCK = None


def acquire_runner_lock(records_dir: Path) -> bool:
    """Take the records directory's runner lock for the lifetime of this process.

    The lock file is opened without truncation and never deleted; whether a runner is active is
    decided by the OS lock alone, so a file left by a dead runner needs no cleanup and two runners
    starting together cannot both get it. Diagnostics go to .runner.info.json (not used to decide).
    """
    global _RUNNER_LOCK
    fh = open(records_dir / ".runner.lock", "a+b")
    if not _try_lock(fh):
        fh.close()
        return False
    _RUNNER_LOCK = fh
    (records_dir / ".runner.info.json").write_text(json.dumps({
        "pid": os.getpid(), "hostname": socket.gethostname(),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}), encoding="utf-8")
    return True


def _maybe_crash(stage: str, aid: str) -> None:
    """Test hook: SCRIPTED_OFFER_TEST_CRASH=<stage>[:<attempt_id>] kills this process at that point."""
    spec = os.environ.get("SCRIPTED_OFFER_TEST_CRASH", "")
    if spec and spec.split(":", 1)[0] == stage and (":" not in spec or spec.split(":", 1)[1] == aid):
        print(f"[test] crashing at {stage} ({aid})", flush=True)
        os._exit(17)


# ----------------------------------------------------------------------
# Reading the shared files
# ----------------------------------------------------------------------
def _jsonl(path: Path):
    """Rows of a JSONL file; unparseable lines (e.g. cut off by a crash) are skipped."""
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _event_key(row):
    return (row.get("stream"), row.get("attempt_id"), row.get("event_seq"))


class SharedState:
    """What the parent knows about the records directory: used attempt numbers, logged attempts and
    merged event keys. Valid because only the lock holder writes to the directory."""

    def __init__(self, records_dir: Path, series: int = 1):
        self.records_dir = records_dir
        self.series = series
        self.max_attempt = defaultdict(int)
        self.logged = set()
        legacy = defaultdict(int)
        for r in _jsonl(records_dir / "run_log.jsonl"):
            p = parse_attempt_id(r.get("attempt_id"))
            if p:
                self.logged.add(r["attempt_id"])
                self._see(p)
            elif "condition" in r and "game_number" in r:
                legacy[(r["condition"], r["game_number"])] += 1   # lines written before attempt ids
        for k, n in legacy.items():
            self.max_attempt[k] = max(self.max_attempt[k], n)
        self.keys = {}
        for stream, name in STREAM_FILES.items():
            rows = _jsonl(records_dir / name)
            self.keys[stream] = {_event_key(r) for r in rows if r.get("attempt_id")}
            for r in rows:
                p = parse_attempt_id(r.get("attempt_id"))
                if p:
                    self._see(p)
        for base in ("_workers", "_attempts"):
            d = records_dir / base
            for sub in (sorted(d.iterdir()) if d.exists() else []):
                for aid in (sub.name, _attempt_json(sub).get("attempt_id")):
                    p = parse_attempt_id(aid)
                    if p:
                        self._see(p)

    def _see(self, parsed):
        cond, g, n = parsed
        self.max_attempt[(cond, g)] = max(self.max_attempt[(cond, g)], n)

    def next_attempt(self, cond: str, g: int) -> str:
        n = self.max_attempt[(cond, g)] + 1
        self.max_attempt[(cond, g)] = n
        return attempt_id(cond, g, n)


def _attempt_json(adir: Path) -> dict:
    try:
        return json.loads((adir / "attempt.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


# ----------------------------------------------------------------------
# Reservation, logs, merging
# ----------------------------------------------------------------------
def reserve_attempt(records_dir: Path, aid: str):
    """Create _workers/<aid>/ and its attempt.json, both exclusively. Returns (dir, None) or (None, error).

    On failure nothing that existed before is touched; only a directory created by this call, and
    still empty, is removed (os.rmdir fails on a non-empty directory).
    """
    workers = records_dir / "_workers"
    workers.mkdir(parents=True, exist_ok=True)
    adir = workers / aid
    try:
        os.mkdir(adir)
    except OSError as e:
        return None, f"attempt reservation failed: {adir}: {e!r}"
    try:
        fd = os.open(adir / "attempt.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        try:
            os.rmdir(adir)
        except OSError:
            pass
        return None, f"attempt reservation failed: {adir / 'attempt.json'}: {e!r}"
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"attempt_id": aid, "runner_pid": os.getpid(),
                   "reserved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    return adir, None


def preserve_old_log(records_dir: Path, cond: str, g: int) -> None:
    """Rename an existing log of this game to <cond>_g<NN>.attempt<n>.log (n from its header)."""
    logs = records_dir / "logs"
    log = logs / f"{cond}_g{g:02d}.log"
    if not log.exists():
        return
    with open(log, encoding="utf-8", errors="replace") as fh:
        first = fh.readline().strip()
    p = parse_attempt_id(first[len("# attempt_id="):]) if first.startswith("# attempt_id=") else None
    stem = f"{cond}_g{g:02d}.attempt{p[2]}" if p else f"{cond}_g{g:02d}.attempt-legacy"
    target, k = logs / f"{stem}.log", 1
    while target.exists():
        k += 1
        target = logs / f"{stem}.{k}.log"
    os.rename(log, target)


def _merge_record(records_dir: Path, adir: Path, cond: str, g: int, aid=None, series: int = 1):
    """Move an attempt's record(s) into records_dir after checking them. Returns an error message or None.

    A series attempt has `series` records (game 1..k); all are checked before any is moved, and a
    series is merged only as a whole.
    """
    moves = []
    for i in range(1, series + 1):
        name = record_name(cond, g, series, i)
        src, dst = adir / name, records_dir / name
        if not src.exists():
            return f"record missing: {src}"
        try:
            rec = json.loads(src.read_text(encoding="utf-8"))
            so = rec["scripted_offer"]
        except Exception as e:
            return f"record unreadable: {src}: {e!r}"
        expected = (cond, g, series_seed(g, i) if series > 1 else g)
        if (so.get("condition"), so.get("game_number"), so.get("seed")) != expected:
            return f"record mismatch: {src} has {(so.get('condition'), so.get('game_number'), so.get('seed'))}"
        if series > 1 and (so.get("series") or {}).get("index") != i:
            return f"record mismatch: {src} is not game {i} of the series"
        if aid is not None and so.get("attempt_id") != aid:
            return f"record mismatch: {src} has attempt_id {so.get('attempt_id')!r}, expected {aid!r}"
        if dst.exists():
            return f"duplicate record: {dst} already exists"
        moves.append((src, dst))
    for src, dst in moves:
        os.replace(src, dst)
    return None


def merge_events(state: SharedState, adir: Path, aid: str) -> bool:
    """Append the attempt's event rows that are not in records_dir yet; True if all are there after."""
    ok = True
    for stream, name in STREAM_FILES.items():
        rows = [r for r in _jsonl(adir / name) if r.get("attempt_id") == aid]
        if not rows:
            continue
        dst = state.records_dir / name
        keys = state.keys[stream]
        new = [r for r in rows if _event_key(r) not in keys]
        if new:
            start = dst.stat().st_size if dst.exists() else 0
            with open(dst, "a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                os.fsync(f.fileno())
            with open(dst, encoding="utf-8") as f:
                f.seek(start)
                written = {_event_key(json.loads(l)) for l in f.read().splitlines() if l.strip()}
            keys |= written
        ok = ok and all(_event_key(r) in keys for r in rows)
    return ok


def api_summary(adir: Path, aid: str) -> dict:
    ev = [r for r in _jsonl(adir / STREAM_FILES["api_error"]) if r.get("attempt_id") == aid]
    raised = [r for r in ev if r.get("action") == "raise"]
    return {"api_errors": len(ev), "api_retries": sum(r.get("action") == "retry" for r in ev),
            "api_error": ({k: raised[-1].get(k) for k in ("player", "model", "error_type", "status_code",
                                                          "transient", "message")} if raised else None)}


def write_run_log(state: SharedState, line: dict) -> None:
    if line["attempt_id"] in state.logged:
        return
    _append_line(state.records_dir / "run_log.jsonl", line)
    state.logged.add(line["attempt_id"])


def archive_attempt(state: SharedState, adir: Path) -> bool:
    """Move a finished attempt directory to _attempts/ (nothing is deleted). False if not moved."""
    dest = state.records_dir / "_attempts" / adir.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"WARNING: {dest} exists; leaving {adir} in place")
        return False
    os.replace(adir, dest)
    return True


def finish_attempt(state: SharedState, line: dict) -> dict:
    """Parent-side completion of one attempt: record, run_log line, events, archive (in that order)."""
    adir, aid = Path(line.pop("attempt_dir")), line["attempt_id"]
    cond, g = line["condition"], line["game_number"]
    if line["status"] == "ok":
        problem = _merge_record(state.records_dir, adir, cond, g, aid, state.series)
        if problem:
            line["status"], line["error"] = "error", problem
    _maybe_crash("after_record_merge", aid)
    line.update(api_summary(adir, aid))
    write_run_log(state, line)
    _maybe_crash("after_runlog", aid)
    merged = merge_events(state, adir, aid)
    _maybe_crash("after_event_merge", aid)
    if merged:
        archive_attempt(state, adir)
    else:
        print(f"WARNING: events of {aid} could not be verified; leaving {adir} in place")
    return line


def recover_attempts(state: SharedState):
    """Merge attempt directories left behind by a runner that died. Returns games not to queue now.

    Called only by the lock holder, so the runner that made them is gone. A directory whose
    worker lock is still held belongs to a live orphaned worker and is left untouched.
    """
    busy = set()
    workers = state.records_dir / "_workers"
    for adir in (sorted(p for p in workers.iterdir() if p.is_dir()) if workers.exists() else []):
        aid = _attempt_json(adir).get("attempt_id") or adir.name
        p = parse_attempt_id(aid)
        if not p:
            print(f"WARNING: unrecognised directory {adir}; left in place")
            continue
        cond, g, _ = p
        wl = adir / "worker.lock"
        if wl.exists() and not _lock_is_free(wl):
            print(f"WARNING: {aid} is still running in another process; left in place, game not queued")
            busy.add((cond, g))
            continue
        rec_path = state.records_dir / record_name(cond, g, state.series)
        recovered_ok = False
        if rec_path.exists():
            try:
                recovered_ok = json.loads(rec_path.read_text(encoding="utf-8"))["scripted_offer"].get("attempt_id") == aid
            except Exception:
                recovered_ok = False
        line = {"condition": cond, "game_number": g, "seed": g, "attempt_id": aid, "job_index": None,
                "execution_position": None, "execution_order": None,
                "status": "ok" if recovered_ok else "error", "seconds": None, "rounds": None, "winner": None,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "error": None if recovered_ok else "interrupted", "recovered": True, "pid": None}
        line.update(api_summary(adir, aid))
        write_run_log(state, line)
        if merge_events(state, adir, aid):
            archive_attempt(state, adir)
        print(f"recovered {aid}: {line['status']}{'' if recovered_ok else ' (interrupted)'}")
    return busy


# ----------------------------------------------------------------------
# One attempt (runs in a pool process, or in this process with --workers 1)
# ----------------------------------------------------------------------
def run_attempt(job: dict) -> dict:
    cond, g, aid, opts = job["condition"], job["game_number"], job["attempt_id"], job["opts"]
    records_dir, adir = Path(opts["records_dir"]), Path(job["attempt_dir"])
    t0 = time.time()
    status, err, record = "ok", None, None
    lock = None
    try:
        lock = open(adir / "worker.lock", "a+b")
        if not _try_lock(lock):
            lock.close()
            lock = None
            raise RuntimeError(f"worker lock of {aid} is held by another process")
        os.chdir(ROOT)
        if opts["mock"]:
            _install_mock(opts.get("mock_fault"), cond, g)
        else:
            _configure_router_env(opts["router"])
            _install_usage_tracking()
        from src.game.game_scripted_offer import GameScriptedOffer
        _usage_ctx.update(condition=cond, game_number=g, attempt_id=aid, seq=0,
                          usage_file=str(adir / STREAM_FILES["usage"]),
                          events_file=str(adir / STREAM_FILES["api_error"]))
        log_path = records_dir / "logs" / f"{cond}_g{g:02d}.log"
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            print(f"# attempt_id={aid}")
            try:
                series = int(opts.get("series") or 1)
                if series == 1:
                    game = GameScriptedOffer(cond, g, seed=g, out_dir=str(adir), offer_round=opts["offer_round"],
                                             bluff_prob=opts["bluff_prob"], attempt_id=aid)
                    # Upstream GameRecord auto-saves to a second-resolution timestamp name; keep it per attempt.
                    game.game_record.save_directory = str(adir)
                    record = game.start_game()
                else:
                    # HB1 series: games 1..k in this process, one after the other. Game i > 1 carries a
                    # memory block per agent built from the records of games 1..i-1 of this attempt.
                    from src.game.game_scripted_offer import AGENT_CONFIGS, SERIES_MEMORY_TEMPLATE
                    from src.utils.prompt_loader import PromptLoader
                    template = PromptLoader.load_prompt(SERIES_MEMORY_TEMPLATE)
                    earlier = []
                    for i in range(1, series + 1):
                        memory = ({c["name"]: build_memory_block(earlier, c["name"], i, template)
                                   for c in AGENT_CONFIGS} if i > 1 else None)
                        print(f"# series game {i}/{series}, seed {series_seed(g, i)}")
                        game = GameScriptedOffer(cond, g, seed=series_seed(g, i), out_dir=str(adir),
                                                 offer_round=opts["offer_round"], bluff_prob=opts["bluff_prob"],
                                                 attempt_id=aid, series_index=i, series_games=series,
                                                 memory_blocks=memory)
                        game.game_record.save_directory = str(adir)
                        record = game.start_game()
                        earlier.append(record)
            except Exception:
                status, err = "error", traceback.format_exc()
                print(err)
    except Exception:
        status, err = "error", traceback.format_exc()
    finally:
        _usage_ctx.update(usage_file=None, events_file=None)
        if lock is not None:
            _unlock(lock)
    return {
        "condition": cond, "game_number": g, "seed": g, "attempt_id": aid, "job_index": job["job_index"],
        "execution_position": job["execution_position"], "execution_order": job["execution_order"],
        "status": status, "seconds": round(time.time() - t0, 1),
        "rounds": len(record["rounds"]) if record else None,
        "winner": record.get("winner") if record else None,
        "mock": opts["mock"],
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error": err, "workers": opts["workers"], "pid": os.getpid(), "attempt_dir": job["attempt_dir"],
    }


def build_jobs(state: SharedState, conditions, games: int, busy, opts):
    """The job queue in counterbalanced order, with attempt ids fixed now (by the lock holder)."""
    jobs = []
    for g in range(games):
        order = execution_order(conditions, g)
        for position, cond in enumerate(order):
            name = record_name(cond, g, state.series)
            if (state.records_dir / name).exists():
                print(f"skip {name} (exists)")
                continue
            if (cond, g) in busy:
                print(f"skip {name} (an earlier attempt is still running)")
                continue
            jobs.append({"condition": cond, "game_number": g, "execution_position": position,
                         "execution_order": order, "attempt_id": state.next_attempt(cond, g), "opts": opts})
    pairs = [(j["condition"], j["game_number"]) for j in jobs]
    assert len(pairs) == len(set(pairs)), "a game was queued twice"
    for i, j in enumerate(jobs):
        j["job_index"] = i
    return jobs


def _prepare(state: SharedState, job: dict):
    """Reserve the attempt and keep any earlier log. Returns an error run_log line if reservation fails."""
    adir, problem = reserve_attempt(state.records_dir, job["attempt_id"])
    if problem:
        return {k: job[k] for k in ("condition", "game_number", "attempt_id", "job_index",
                                    "execution_position", "execution_order")} | {
            "seed": job["game_number"], "status": "error", "seconds": None, "rounds": None, "winner": None,
            "mock": job["opts"]["mock"], "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": problem, "pid": None, "api_errors": 0, "api_retries": 0, "api_error": None}
    job["attempt_dir"] = str(adir)
    _maybe_crash("after_reserve", job["attempt_id"])
    preserve_old_log(state.records_dir, job["condition"], job["game_number"])
    return None


def _report(line: dict) -> None:
    print(f"{line['condition']} g{line['game_number']:02d}: {line['status']} in {line['seconds']}s, "
          f"rounds={line['rounds']}, winner={line['winner']}, pid={line['pid']}, attempt={line['attempt_id']}"
          + (f", api_retries={line['api_retries']}" if line.get("api_retries") else "")
          + (f", error={line['error'].strip().splitlines()[-1]}" if line["status"] != "ok" and line["error"] else ""),
          flush=True)


def run_jobs(state: SharedState, jobs, workers: int) -> int:
    failures = 0

    def done(line):
        nonlocal failures
        line = finish_attempt(state, line)
        _report(line)
        failures += line["status"] != "ok"

    if workers <= 1:
        for job in jobs:
            bad = _prepare(state, job)
            if bad:
                write_run_log(state, bad)
                _report(bad)
                failures += 1
                continue
            done(run_attempt(job))
        return failures

    import multiprocessing as mp
    results = queue.Queue()
    pending = iter(jobs)
    in_flight = 0
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        def submit_next() -> bool:
            nonlocal failures
            for job in pending:
                bad = _prepare(state, job)
                if bad:
                    write_run_log(state, bad)
                    _report(bad)
                    failures += 1
                    continue
                pool.apply_async(run_attempt, (job,), callback=results.put,
                                 error_callback=lambda e, job=job: results.put(
                                     {**{k: job[k] for k in ("condition", "game_number", "attempt_id", "job_index",
                                                             "execution_position", "execution_order",
                                                             "attempt_dir")},
                                      "seed": job["game_number"], "status": "error", "seconds": None,
                                      "rounds": None, "winner": None, "mock": job["opts"]["mock"],
                                      "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      "error": repr(e), "workers": workers, "pid": None}))
                return True
            return False

        while True:
            while in_flight < workers and submit_next():
                in_flight += 1
            if in_flight == 0:
                break
            line = results.get()   # completion order
            in_flight -= 1
            done(line)
        pool.close()
        pool.join()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Scripted-offer experiment runner")
    ap.add_argument("--conditions", default="C0,C1,C1b,C2,C3,C4,C5")
    ap.add_argument("--games", type=int, default=12, help="game numbers 0..games-1 per condition")
    ap.add_argument("--series", type=int, default=1,
                    help="HB1 only (--conditions C1s): k consecutive games per game number with true memory "
                         "from game 2 on; records <cond>_gNN_kI.json. Default 1 = single games.")
    ap.add_argument("--pilot", action="store_true", help="one game (game number 0) per condition")
    ap.add_argument("--offer-round", type=int, default=2)
    ap.add_argument("--bluff-prob", type=float, default=0.5)
    ap.add_argument("--records-dir", default=str(DEFAULT_RECORDS))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--router", default="http://127.0.0.1:8100")
    ap.add_argument("--mock", action="store_true", help="offline: use MockLLMClient, no network")
    ap.add_argument("--mock-fault", default=None,
                    help="tests only, with --mock: COND:G:SITE:KIND[:N[:SKIP]] (see src/test/mock_llm.py)")
    ap.add_argument("--workers", type=int, default=None,
                    help="games run concurrently, one fresh process per game; real (non --mock) runs need >= 2 "
                         "(1 = in this process, --mock only). Default: 10, or 1 with --mock")
    ap.add_argument("--warmup-routes", default=None,
                    help="routes.json of the model backends: wake all of them concurrently (direct, not via "
                         "the router; nothing is logged in the records) before the first game. Ignored with --mock.")
    ap.add_argument("--warmup-script", default=str(DEFAULT_WARMUP_SCRIPT),
                    help="path of experiments/modal_infra/warmup.py")
    args = ap.parse_args()
    if args.mock_fault and not args.mock:
        ap.error("--mock-fault requires --mock")
    if args.workers is None:
        args.workers = 1 if args.mock else 10
    if args.workers < 2 and not args.mock:
        ap.error("real runs are parallel only: --workers must be >= 2 (serial --workers 1 is allowed with --mock)")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.series < 1:
        ap.error("--series must be >= 1")
    if args.series > 1 and conditions != ["C1s"]:
        ap.error("--series is defined for --conditions C1s only (one condition per records directory)")
    if args.series == 1 and "C1s" in conditions:
        ap.error("C1s needs --series k (k >= 2)")
    games = 1 if args.pilot else args.games
    records_dir = Path(args.records_dir).resolve()   # games chdir to ROOT
    (records_dir / "logs").mkdir(parents=True, exist_ok=True)
    if not acquire_runner_lock(records_dir):
        print(f"another runner holds {records_dir / '.runner.lock'}; not starting", flush=True)
        return EXIT_LOCKED
    _copy_verbatim(Path(args.results_dir))
    if args.warmup_routes and not args.mock:
        if not warm_up(Path(args.warmup_script), Path(args.warmup_routes), records_dir / "warmup"):
            print("warm-up failed: not starting any game", flush=True)
            return EXIT_WARMUP
        print("warm-up complete: every backend answered", flush=True)

    state = SharedState(records_dir, series=args.series)
    busy = recover_attempts(state)
    opts = {"records_dir": str(records_dir), "mock": args.mock, "mock_fault": args.mock_fault,
            "router": args.router, "offer_round": args.offer_round, "bluff_prob": args.bluff_prob,
            "workers": args.workers, "series": args.series}
    jobs = build_jobs(state, conditions, games, busy, opts)
    failures = run_jobs(state, jobs, args.workers)
    return 1 if failures or busy else 0


if __name__ == "__main__":
    sys.exit(main())
