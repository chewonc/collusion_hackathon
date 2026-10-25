"""Offline checks of API-error handling, attempts, locks and recovery (no network; mock transport).

Covers: StrictLLMClient (backoff, transient vs fatal, no SDK retries, identical requests); API errors
never becoming fallbacks in play / challenge / offer / reflect; the runner logging status=error,
merging no record and keeping every API error and usage row across attempts; attempt numbering and
reservation; the runner lock (live runner, simultaneous start, files left by a dead runner, orphaned
workers); crash recovery without duplicated events; completion-order merging.

Run from Collusion_Liars_Bar/:
    ../.venv/Scripts/python.exe src/test/test_api_errors.py
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import openai  # noqa: E402
import httpx2  # noqa: E402

from src.agents import player as player_module  # noqa: E402
from src.agents import experiment_player as ep  # noqa: E402
from src.llm import llm_client as llm_module  # noqa: E402
from src.test.mock_llm import MockFault, MockLLMClient, MockTransport, _Obj  # noqa: E402

player_module.LLMClient = MockLLMClient
llm_module.resolve_model = lambda m, *a, **k: m

from src.game import run_scripted_offer as rso  # noqa: E402
from src.game.game_scripted_offer import GameScriptedOffer  # noqa: E402

PY = sys.executable
RUNNER = "src/game/run_scripted_offer.py"
ENV = {**os.environ, "PYTHONUTF8": "1"}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _req():
    return httpx2.Request("POST", "http://mock.invalid/v1/chat/completions")


def _status_error(cls, code):
    return cls(f"mock {code}", response=httpx2.Response(code, request=_req()), body=None)


class Scripted:
    """Fake transport: raises the given exceptions in order, then answers `text`; records kwargs."""

    def __init__(self, errors, text='{"was_challenged": false, "challenge_reason": "x"}', choices=True):
        self.errors, self.text, self.calls, self.choices = list(errors), text, [], choices
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        if not self.choices:
            return _Obj(choices=[], usage=None)
        return _Obj(choices=[_Obj(message=_Obj(content=self.text))], usage=None)


def run(records, *args, env=None, check=None):
    cmd = [PY, RUNNER, "--mock", "--records-dir", str(records), "--results-dir", str(Path(records).parent / "res"),
           *args]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env or ENV)
    if check is not None:
        assert out.returncode == check, f"exit {out.returncode} != {check}\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    return out


def popen(records, *args, env=None):
    cmd = [PY, RUNNER, "--mock", "--records-dir", str(records), "--results-dir", str(Path(records).parent / "res"),
           *args]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env or ENV)


def jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def tree_hash(d):
    """File list with content hashes; lock files (byte-locked while held) are compared by presence only."""
    d = Path(d)

    def h(p):
        if not p.is_file():
            return "dir"
        if p.suffix == ".lock":
            return "lock"
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return {str(p.relative_to(d)): h(p) for p in sorted(d.rglob("*"))}


def wait_for(cond, timeout=60, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in out.stdout


# ----------------------------------------------------------------------
# 1. StrictLLMClient
# ----------------------------------------------------------------------
def test_strict_client():
    events = []
    ep.set_api_event_sink(events.append)
    try:
        sleeps = []
        t = Scripted([openai.APIConnectionError(message="x", request=_req()),
                      openai.APITimeoutError(request=_req())])
        c = ep.StrictLLMClient(client=t, sleep=sleeps.append)
        out, _ = c.chat([{"role": "user", "content": "hi"}], model="qwen2.5-7b")
        assert json.loads(out)["was_challenged"] is False
        assert c.api_retries == 2 and sleeps == [2, 4] and len(t.calls) == 3
        assert [e["action"] for e in events] == ["retry", "retry"]

        events.clear()
        t = Scripted([openai.APIConnectionError(message="x", request=_req())] * 100)
        c = ep.StrictLLMClient(client=t, sleep=lambda s: None)
        try:
            c.chat([{"role": "user", "content": "hi"}], model="qwen2.5-7b")
            raise AssertionError("expected ApiTransientExhausted")
        except ep.ApiTransientExhausted as e:
            assert isinstance(e.__cause__, openai.APIConnectionError)
        n = len(ep.StrictLLMClient.BACKOFF_S)
        assert c.api_retries == n and len(t.calls) == n + 1
        assert sum(ep.StrictLLMClient.BACKOFF_S) >= 600, "backoff budget should be about 10 minutes"
        assert [e["action"] for e in events] == ["retry"] * n + ["raise"]

        for exc, transient in ((_status_error(openai.AuthenticationError, 401), False),
                               (_status_error(openai.BadRequestError, 400), False),
                               (_status_error(openai.NotFoundError, 404), False),
                               (_status_error(openai.RateLimitError, 429), True),
                               (_status_error(openai.InternalServerError, 500), True),
                               (_status_error(openai.APIStatusError, 503), True),
                               (ValueError("not an API exception"), False)):
            assert ep.is_transient(exc) == transient, exc
            if not transient:
                t = Scripted([exc])
                c = ep.StrictLLMClient(client=t, sleep=lambda s: None)
                try:
                    c.chat([{"role": "user", "content": "hi"}], model="qwen2.5-7b")
                    raise AssertionError("expected ApiFatalError")
                except ep.ApiFatalError:
                    pass
                assert c.api_retries == 0 and len(t.calls) == 1, "fatal errors are not retried"

        for bad in (Scripted([], choices=False),):
            try:
                ep.StrictLLMClient(client=bad, sleep=lambda s: None).chat([{"role": "user", "content": "x"}],
                                                                         model="qwen2.5-7b")
                raise AssertionError("expected ApiFatalError on a malformed response")
            except ep.ApiFatalError:
                pass
        try:
            ep.StrictLLMClient(client=Scripted([]), sleep=lambda s: None).chat([], model="no-such-model")
            raise AssertionError("expected ApiFatalError on an unsupported model")
        except ep.ApiFatalError:
            pass

        # Requests are exactly LLMClient's (same model id, sampling, token cap, extra_body).
        msgs = [{"role": "user", "content": "same prompt"}]
        for model in ("llama-3.1-8b", "llama-3.2-3b", "qwen2.5-7b"):
            a, b = Scripted([]), Scripted([])
            base = llm_module.LLMClient.__new__(llm_module.LLMClient)
            base._client = a
            with contextlib.redirect_stdout(io.StringIO()):
                ra, _ = base.chat(msgs, model=model)
                rb, _ = ep.StrictLLMClient(client=b).chat(msgs, model=model)
            assert a.calls == b.calls and ra == rb, f"{model}: request or cleaned response differs"

        # No SDK-internal retries for the real client.
        orig = llm_module.make_client
        seen = {}
        llm_module.make_client = lambda **kw: seen.update(kw) or Scripted([])
        try:
            ep.StrictLLMClient()
        finally:
            llm_module.make_client = orig
        assert seen == {"max_retries": 0}
    finally:
        ep.set_api_event_sink(None)
    print("ok StrictLLMClient: backoff on transient errors, immediate fatal errors, exhausted -> raise, "
          "SDK retries off, requests identical to LLMClient")


# ----------------------------------------------------------------------
# 2. Players: an API error is re-raised, never a fallback
# ----------------------------------------------------------------------
def test_players_reraise():
    for exc_cls in (ep.ApiFatalError, ep.ApiTransientExhausted):
        class Raising:
            def chat(self, messages, model=None, max_new_tokens=4096):
                raise exc_cls("mock")
        ep.set_strict_client_factory(Raising)
        try:
            p = ep.ExperimentPlayer("Lily", "llama-3.1-8b")
            p.hand = ["A", "K"]
            for name, call in (("play", lambda: p.choose_cards_to_play("b", "a", "d")),
                               ("challenge", lambda: p.decide_challenge("b", "a", "d", "perf", "")),
                               ("reflect", lambda: p.reflect(["Lily", "Luke"], "b", "a", "r"))):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        call()
                    raise AssertionError(f"{name}: {exc_cls.__name__} was swallowed")
                except ep.ApiRequestError:
                    pass
            assert not p.last_play_fallback and not p.last_challenge_fallback
            assert p.hand == ["A", "K"], "a failed play must not touch the hand"
        finally:
            ep.set_strict_client_factory(None)

    # Transient errors that recover: normal answer, no fallback, retries counted.
    ep.set_strict_client_factory(lambda: ep.StrictLLMClient(
        client=Scripted([openai.APIConnectionError(message="x", request=_req())] * 2), sleep=lambda s: None))
    try:
        p = ep.ExperimentPlayer("Lily", "llama-3.1-8b")
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = p.decide_challenge("b", "a", "d", "perf", "")
        assert res["was_challenged"] is False and "is_fallback" not in res and not p.last_challenge_fallback
        assert p.api_retries == 2
    finally:
        ep.set_strict_client_factory(None)
    print("ok players: ApiFatalError / ApiTransientExhausted re-raised from play, challenge and reflect "
          "(no fallback, hand intact); recovered transient errors give the normal answer")


# ----------------------------------------------------------------------
# 3. Retries do not change the game
# ----------------------------------------------------------------------
def _direct_game(fault_spec, tmp):
    fault = MockFault(fault_spec)
    ep.set_strict_client_factory(lambda: ep.StrictLLMClient(client=MockTransport(fault), sleep=lambda s: None))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rec = GameScriptedOffer("C1", 0, seed=0, out_dir=str(tmp)).start_game()
    finally:
        ep.set_strict_client_factory(None)
    rec.pop("game_id", None)
    return rec


def test_retries_do_not_change_game(tmp):
    base = _direct_game("C1:0:any:transient:0", tmp)
    for site in ("play", "challenge", "offer", "reflect"):
        skip = 1 if site == "offer" else 3   # only three offer calls per game
        rec = _direct_game(f"C1:0:{site}:transient:2:{skip}", tmp)
        retries = rec["scripted_offer"].pop("api_retries")
        assert sum(retries.values()) == 2, (site, retries)
        b = dict(base["scripted_offer"])
        b.pop("api_retries")
        assert {**rec, "scripted_offer": rec["scripted_offer"]} == {**base, "scripted_offer": b}, \
            f"{site}: a game with recovered API errors differs from the same game without them"
    print("ok two recovered API errors at play / challenge / offer / reflect leave the game record unchanged")


# ----------------------------------------------------------------------
# 4. Runner: failed attempts, records, logs, events, rerun
# ----------------------------------------------------------------------
def test_runner_failures(tmp):
    for workers in ("1", "2"):
        for site in ("play", "challenge", "offer", "reflect"):
            for kind, exc_name in (("fatal", "ApiFatalError"), ("transient", "ApiTransientExhausted")):
                rec = tmp / f"fail_{workers}_{site}_{kind}" / "rec"
                out = run(rec, "--games", "1", "--conditions", "C1,C4", "--workers", workers,
                          "--mock-fault", f"C1:0:{site}:{kind}", check=1)
                log = jsonl(rec / "run_log.jsonl")
                line = next(r for r in log if r["condition"] == "C1")
                assert line["status"] == "error" and exc_name in line["error"], (site, kind, line["error"][-300:])
                assert line["attempt_id"] == "C1_g00_a1"
                assert next(r for r in log if r["condition"] == "C4")["status"] == "ok"
                assert not (rec / "C1_g00.json").exists(), "a failed attempt must not produce a normal record"
                assert (rec / "C4_g00.json").exists()
                ev = jsonl(rec / "api_errors.jsonl")
                assert ev and all(e["attempt_id"] == "C1_g00_a1" for e in ev) and ev[-1]["action"] == "raise"
                assert line["api_error"]["error_type"] == ev[-1]["error_type"]
                assert line["api_retries"] == sum(e["action"] == "retry" for e in ev)
                assert any(u["attempt_id"] == "C1_g00_a1" for u in jsonl(rec / "usage.jsonl"))
                text = (rec / "logs" / "C1_g00.log").read_text(encoding="utf-8")
                after = text[text.index("LLM API error"):]
                assert "defaulting to" not in after, f"{site}/{kind}: a fallback followed the API error"
                assert not list((rec / "_workers").iterdir()), "attempt directory not archived"
                assert (rec / "_attempts" / "C1_g00_a1").is_dir()
    print("ok failed attempts (fatal and exhausted transient, at play/challenge/offer/reflect, 1 and 2 workers): "
          "status=error with the API exception, no record, no fallback, events and usage merged, dir archived")

    # Rerun without the fault: only that game, as attempt 2; attempt 1's log and events kept.
    rec = tmp / "fail_1_play_fatal" / "rec"
    a1_events = jsonl(rec / "api_errors.jsonl")
    out = run(rec, "--games", "1", "--conditions", "C1,C4", check=0)
    assert "skip C4_g00.json (exists)" in out.stdout
    log = jsonl(rec / "run_log.jsonl")
    assert [(r["attempt_id"], r["status"]) for r in log if r["condition"] == "C1"] == \
        [("C1_g00_a1", "error"), ("C1_g00_a2", "ok")]
    assert json.loads((rec / "C1_g00.json").read_text(encoding="utf-8"))["scripted_offer"]["attempt_id"] == "C1_g00_a2"
    assert (rec / "logs" / "C1_g00.attempt1.log").read_text(encoding="utf-8").startswith("# attempt_id=C1_g00_a1\n")
    assert (rec / "logs" / "C1_g00.log").read_text(encoding="utf-8").startswith("# attempt_id=C1_g00_a2\n")
    assert jsonl(rec / "api_errors.jsonl") == a1_events, "attempt 1's API errors must be kept as they were"
    an = tmp / "an_rerun"
    subprocess.run([PY, "analysis/scripted_offer/analyze.py", "--records", str(rec), "--out", str(an)],
                   cwd=ROOT, capture_output=True, text=True, env=ENV, check=True)
    games = json.loads((an / "usage_summary.json").read_text(encoding="utf-8"))
    summary = (an / "summary.md").read_text(encoding="utf-8")
    assert "Games analysed: 2 " in summary, "the failed attempt must not be analysed"
    print("ok rerun: only the failed game reruns, as C1_g00_a2; attempt-1 log and API errors kept; "
          "analysis reads only completed records")

    # Transient errors that recover inside the runner.
    rec = tmp / "recover" / "rec"
    run(rec, "--games", "1", "--conditions", "C1", "--mock-fault", "C1:0:challenge:transient:3:2", check=0)
    line = jsonl(rec / "run_log.jsonl")[0]
    r = json.loads((rec / "C1_g00.json").read_text(encoding="utf-8"))["scripted_offer"]
    assert line["status"] == "ok" and line["api_retries"] == 3 and sum(r["api_retries"].values()) == 3
    print("ok recovered transient errors: game ok, api_retries = 3 in run_log and in the record")

    # Events of a failed attempt are in records_dir right after it ends (crash before the next game).
    rec = tmp / "early" / "rec"
    env = {**ENV, "SCRIPTED_OFFER_TEST_CRASH": "after_reserve:C4_g00_a1"}
    out = run(rec, "--games", "1", "--conditions", "C1,C4", "--mock-fault", "C1:0:play:fatal", env=env)
    assert out.returncode == 17, out.stdout[-2000:]
    assert any(e["attempt_id"] == "C1_g00_a1" for e in jsonl(rec / "api_errors.jsonl"))
    assert any(r["attempt_id"] == "C1_g00_a1" for r in jsonl(rec / "run_log.jsonl"))
    print("ok a failed attempt's events and run_log line are in records_dir as soon as it ends")


# ----------------------------------------------------------------------
# 5. Attempt numbers, reservation, event keys
# ----------------------------------------------------------------------
def test_numbering_and_reservation(tmp):
    d = tmp / "num"
    (d / "_workers" / "C1_g00_a2").mkdir(parents=True)
    with open(d / "run_log.jsonl", "w", encoding="utf-8") as f:
        for aid in ("C1_g00_a1", "C1_g00_a3"):
            f.write(json.dumps({"condition": "C1", "game_number": 0, "attempt_id": aid}) + "\n")
        for g in range(40):
            f.write(json.dumps({"condition": "C4", "game_number": g, "attempt_id": f"C4_g{g:02d}_a1"}) + "\n")
        f.write(json.dumps({"condition": "C2", "game_number": 5}) + "\n")   # legacy line, no attempt id
    st = rso.SharedState(d)
    assert st.next_attempt("C1", 0) == "C1_g00_a4", "max over run_log and directories, not a line count"
    assert st.next_attempt("C1", 0) == "C1_g00_a5"
    assert st.next_attempt("C3", 7) == "C3_g07_a1", "other games' lines must not count"
    assert st.next_attempt("C2", 5) == "C2_g05_a2", "legacy lines count as earlier attempts"

    # Reservation: an existing directory is an error and is left exactly as it was.
    d = tmp / "resv"
    adir, err = rso.reserve_attempt(d, "C1_g00_a1")
    assert adir and not err and (d / "_workers").is_dir(), "_workers is created when missing"
    (adir / "keep.txt").write_text("precious", encoding="utf-8")
    before = tree_hash(d)
    adir2, err = rso.reserve_attempt(d, "C1_g00_a1")
    assert adir2 is None and "reservation failed" in err and tree_hash(d) == before
    # attempt.json creation failing after mkdir: only the new, empty directory is removed.
    orig = os.open
    os.open = lambda *a, **k: (_ for _ in ()).throw(FileExistsError("race"))
    try:
        adir3, err = rso.reserve_attempt(d, "C1_g00_a9")
    finally:
        os.open = orig
    assert adir3 is None and not (d / "_workers" / "C1_g00_a9").exists() and tree_hash(d) == before

    # A reservation that fails in the runner's own path: logged as an error, nothing touched.
    d = tmp / "race"
    d.mkdir()
    st = rso.SharedState(d)
    job = {"condition": "C1", "game_number": 0, "execution_position": 0, "execution_order": ["C1"],
           "attempt_id": st.next_attempt("C1", 0), "job_index": 0, "opts": {"mock": True}}
    (d / "_workers" / job["attempt_id"]).mkdir(parents=True)
    (d / "_workers" / job["attempt_id"] / "other.txt").write_text("someone else's", encoding="utf-8")
    before = tree_hash(d)
    bad = rso._prepare(st, job)
    assert bad and bad["status"] == "error" and "reservation failed" in bad["error"] and tree_hash(d) == before

    # build_jobs refuses a duplicated game.
    orig = rso.execution_order
    rso.execution_order = lambda conds, g: ["C1", "C1"]
    try:
        rso.build_jobs(rso.SharedState(tmp / "dup"), ["C1"], 1, set(), {})
        raise AssertionError("a duplicated game must be refused")
    except AssertionError as e:
        assert "queued twice" in str(e)
    finally:
        rso.execution_order = orig
    print("ok attempt numbers: max over run_log/events/dirs per game (a1,a3 + dir a2 -> a4), other games ignored; "
          "reservation never deletes or overwrites; duplicates refused")

    # One event_seq counter per attempt across streams.
    rec = tmp / "keys" / "rec"
    run(rec, "--games", "1", "--conditions", "C1", "--mock-fault", "C1:0:any:transient:4:10", check=0)
    rows = [r for r in jsonl(rec / "usage.jsonl") + jsonl(rec / "api_errors.jsonl") if r["attempt_id"] == "C1_g00_a1"]
    seqs = [r["event_seq"] for r in rows]
    assert {r["stream"] for r in rows} == {"usage", "api_error"} and len(seqs) == len(set(seqs))
    print("ok event_seq is unique across usage and api_error rows of one attempt")


# ----------------------------------------------------------------------
# 6. Crash recovery without duplicates
# ----------------------------------------------------------------------
def test_crash_recovery(tmp):
    fault = ["--mock-fault", "C1:0:any:transient:3:5"]
    for stage, expect_status, expect_rerun in (("after_reserve", "error", True),
                                               ("after_record_merge", "ok", False),
                                               ("after_runlog", "ok", False),
                                               ("after_event_merge", "ok", False)):
        rec = tmp / f"crash_{stage}" / "rec"
        env = {**ENV, "SCRIPTED_OFFER_TEST_CRASH": f"{stage}:C1_g00_a1"}
        out = run(rec, "--games", "1", "--conditions", "C1", *fault, env=env)
        assert out.returncode == 17, (stage, out.stdout[-1500:])
        assert (rec / ".runner.lock").exists() and (rec / "_workers" / "C1_g00_a1").is_dir()
        events_before = jsonl(rec / "api_errors.jsonl") + jsonl(rec / "usage.jsonl")
        run(rec, "--games", "1", "--conditions", "C1", *fault, check=0)
        log = [r for r in jsonl(rec / "run_log.jsonl") if r["attempt_id"] == "C1_g00_a1"]
        assert len(log) == 1 and log[0]["status"] == expect_status, (stage, log)
        if expect_status == "error":
            assert log[0]["error"] == "interrupted"
        reruns = [r for r in jsonl(rec / "run_log.jsonl") if r["attempt_id"] == "C1_g00_a2"]
        assert bool(reruns) == expect_rerun, (stage, reruns)
        a1 = [r for r in jsonl(rec / "api_errors.jsonl") + jsonl(rec / "usage.jsonl") if r["attempt_id"] == "C1_g00_a1"]
        keys = [(r["stream"], r["event_seq"]) for r in a1]
        assert len(keys) == len(set(keys)), f"{stage}: duplicated events after recovery"
        if stage == "after_event_merge":
            assert len(a1) == len([r for r in events_before if r["attempt_id"] == "C1_g00_a1"]), \
                "recovery must not add events that were already merged"
        if stage != "after_reserve":
            assert len(a1) > 0
        assert not list((rec / "_workers").iterdir()) and (rec / "_attempts" / "C1_g00_a1").is_dir()
        rec_id = json.loads((rec / "C1_g00.json").read_text(encoding="utf-8"))["scripted_offer"]["attempt_id"]
        assert rec_id == ("C1_g00_a2" if expect_rerun else "C1_g00_a1")
    print("ok crash after reserve -> interrupted + rerun as a2; after record merge / run_log / event merge -> "
          "completed without rerun; no duplicated run_log lines or events; directories archived")


# ----------------------------------------------------------------------
# 7. Runner lock, orphaned workers, completion order
# ----------------------------------------------------------------------
def test_locks(tmp):
    # A live runner: a second one is refused and touches nothing.
    rec = tmp / "live" / "rec"
    marker = tmp / "live_marker"
    env = {**ENV, "MOCK_HANG_MARKER": str(marker)}
    a = popen(rec, "--games", "1", "--conditions", "C1", "--mock-fault", "C1:0:play:hang:6", env=env)
    wait_for(marker.exists, what="hang marker")
    before = tree_hash(rec / "_workers")
    b = run(rec, "--games", "1", "--conditions", "C1")
    assert b.returncode == rso.EXIT_LOCKED, b.stdout
    assert tree_hash(rec / "_workers") == before
    assert a.wait(120) == 0
    log = jsonl(rec / "run_log.jsonl")
    assert [r["attempt_id"] for r in log] == ["C1_g00_a1"] and not any(r.get("error") == "interrupted" for r in log)
    print("ok a second runner is refused while one is running; it changes nothing")

    # Two runners started together: exactly one gets the lock (5 trials).
    for i in range(5):
        rec = tmp / f"race{i}" / "rec"
        rec.mkdir(parents=True)
        procs = [popen(rec, "--games", "1", "--conditions", "C1", "--mock-fault", "C1:0:play:hang:3")
                 for _ in range(2)]
        codes = sorted(p.wait(120) for p in procs)
        assert codes == [0, rso.EXIT_LOCKED], codes
    print("ok two runners started together: exactly one runs, the other exits with code 3 (5 trials)")

    # Orphaned worker: parent killed, worker alive -> left alone; recovered once it has ended.
    rec = tmp / "orphan" / "rec"
    marker = tmp / "orphan_marker"
    env = {**ENV, "MOCK_HANG_MARKER": str(marker)}
    a = popen(rec, "--games", "1", "--conditions", "C1,C4", "--workers", "2",
              "--mock-fault", "C1:0:play:hang:12", env=env)
    wait_for(marker.exists, what="orphan hang marker")
    worker_pid = int(marker.read_text())
    a.kill()   # TerminateProcess on the parent only; the pool worker keeps running
    a.wait(30)
    assert pid_alive(worker_pid)
    adir = rec / "_workers" / "C1_g00_a1"
    before = tree_hash(adir)
    b = run(rec, "--games", "1", "--conditions", "C1,C4", "--workers", "2")
    assert b.returncode == 1 and "still running" in b.stdout, b.stdout
    assert tree_hash(adir) == before and not (rec / "C1_g00.json").exists()
    assert not any(r["attempt_id"].startswith("C1_g00") for r in jsonl(rec / "run_log.jsonl"))
    wait_for(lambda: not pid_alive(worker_pid), timeout=90, what="orphaned worker to exit")
    c = run(rec, "--games", "1", "--conditions", "C1,C4", "--workers", "2", check=0)
    log = jsonl(rec / "run_log.jsonl")
    assert [(r["attempt_id"], r["status"], r.get("error")) for r in log if r["condition"] == "C1"] == \
        [("C1_g00_a1", "error", "interrupted"), ("C1_g00_a2", "ok", None)], log
    assert (rec / "_attempts" / "C1_g00_a1").is_dir(), "the orphan's directory is archived, not deleted"
    print("ok orphaned worker: not recovered and its game not queued while alive; recovered as interrupted "
          "and rerun as a2 after it ended")

    # Completion order: a slow first job does not hold up the others' merging.
    rec = tmp / "order" / "rec"
    run(rec, "--games", "1", "--workers", "3", "--mock-fault", "C0:0:play:hang:6", check=0)
    log = jsonl(rec / "run_log.jsonl")
    first = next(r for r in log if r["job_index"] == 0)
    assert log[-1] is not None and log.index(first) == len(log) - 1, "job 0 (slow) should be merged last"
    assert all(r["finished_utc"] < first["finished_utc"] for r in log if r is not first)
    seq = tmp / "order_seq" / "rec"
    run(seq, "--games", "1", "--mock-fault", "C0:0:play:hang:0", check=0)
    for p in sorted(seq.glob("C*_g??.json")):
        x = json.loads(p.read_text(encoding="utf-8"))
        y = json.loads((rec / p.name).read_text(encoding="utf-8"))
        x.pop("game_id"), y.pop("game_id")
        assert x == y, f"{p.name}: parallel completion-order run differs from sequential"
    print("ok completion order: other games merged while job 0 hung; records equal to a sequential run")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_strict_client()
        test_players_reraise()
        test_retries_do_not_change_game(tmp)
        test_runner_failures(tmp)
        test_numbering_and_reservation(tmp)
        test_crash_recovery(tmp)
        test_locks(tmp)
    print("ALL API-ERROR CHECKS PASSED")
