"""Runs long-lived deskbot CLI subcommands (research, routines) as real
subprocesses and streams their stdout to the browser over SSE.

Why a subprocess instead of calling run_deep_research()/run_routine()
in-process: those functions print progress via a module-level rich Console
bound to the server's own stdout, and stopping them cleanly needs a real
KeyboardInterrupt delivered to that exact call stack. A subprocess gets both
for free — its stdout is just a pipe we can read line-by-line, and
CTRL_BREAK_EVENT (Windows) / SIGINT (elsewhere) delivered to it triggers the
same graceful-stop code path `deskbot research` already has, with no changes
needed to research.py or routine_runner.py.
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

_REPORT_SAVED_RE = re.compile(r"Report saved to:\s*(.+)")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def deskbot_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "deskbot", *args]


def start_job(cmd: list[str], label: str | None = None) -> str:
    """label is shown as-is in the control panel's Running Jobs list — pass a
    human-readable description (e.g. "Research: creatine (scientist)") since
    the raw command line is not something a non-technical user should have to
    parse to tell two jobs apart. See list_jobs()."""
    job_id = uuid.uuid4().hex
    line_queue: queue.Queue[str | None] = queue.Queue()

    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )

    with _lock:
        _jobs[job_id] = {
            "proc": proc,
            "queue": line_queue,
            "done": False,
            "report_path": None,
            "label": label or " ".join(cmd),
            "started_at": time.time(),
            "pid": proc.pid,
        }

    def _pump() -> None:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            match = _REPORT_SAVED_RE.search(line)
            if match:
                with _lock:
                    _jobs[job_id]["report_path"] = match.group(1).strip()
            line_queue.put(line)
        proc.wait()
        with _lock:
            _jobs[job_id]["done"] = True
        line_queue.put(None)  # sentinel: stream is finished

    threading.Thread(target=_pump, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    """Summary of every job started this server session (running or
    finished), most recently started first — the data behind the control
    panel's Running Jobs panel. This is what the overheating incident was
    missing: a background job started outside any tracked job list had no
    visible way to notice it was still running, let alone stop it. Every job
    started through the web UI now shows up here regardless of which view
    started it, and stays visible after finishing so a stale/crashed job
    doesn't just silently disappear."""
    with _lock:
        jobs = list(_jobs.items())
    now = time.time()
    return [
        {
            "id": job_id,
            "label": job["label"],
            "pid": job["pid"],
            "running": not job["done"],
            "started_at": job["started_at"],
            "elapsed_seconds": round(now - job["started_at"]),
        }
        for job_id, job in sorted(jobs, key=lambda kv: kv[1]["started_at"], reverse=True)
    ]


def stop_job(job_id: str) -> bool:
    job = get_job(job_id)
    if job is None:
        return False
    proc: subprocess.Popen = job["proc"]
    try:
        if sys.platform == "win32":
            import signal

            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass
    return True


def get_report(job_id: str) -> dict | None:
    job = get_job(job_id)
    if job is None or not job.get("report_path"):
        return None
    path = Path(job["report_path"])
    if not path.exists():
        return None
    return {"path": str(path), "content": path.read_text(encoding="utf-8")}
