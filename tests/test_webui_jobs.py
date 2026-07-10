from __future__ import annotations

import sys
import time

from deskbot.webui import jobs


def _quick_cmd(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true within timeout")


def test_start_job_records_label_pid_and_started_at():
    job_id = jobs.start_job(_quick_cmd("print('hi')"), label="test: prints hi")
    job = jobs.get_job(job_id)

    assert job["label"] == "test: prints hi"
    assert job["pid"] == job["proc"].pid
    assert isinstance(job["started_at"], float)
    _wait_until(lambda: jobs.get_job(job_id)["done"])


def test_start_job_without_label_defaults_to_joined_command():
    cmd = _quick_cmd("print('hi')")
    job_id = jobs.start_job(cmd)
    job = jobs.get_job(job_id)

    assert job["label"] == " ".join(cmd)
    _wait_until(lambda: jobs.get_job(job_id)["done"])


def test_list_jobs_reports_running_then_finished():
    job_id = jobs.start_job(_quick_cmd("import time; time.sleep(0.3)"), label="test: sleeper")

    entries = {j["id"]: j for j in jobs.list_jobs()}
    assert job_id in entries
    assert entries[job_id]["running"] is True
    assert entries[job_id]["label"] == "test: sleeper"
    assert entries[job_id]["elapsed_seconds"] >= 0

    _wait_until(lambda: jobs.get_job(job_id)["done"])
    entries = {j["id"]: j for j in jobs.list_jobs()}
    assert entries[job_id]["running"] is False


def test_list_jobs_orders_most_recently_started_first():
    first = jobs.start_job(_quick_cmd("print('a')"), label="test: first")
    time.sleep(0.05)
    second = jobs.start_job(_quick_cmd("print('b')"), label="test: second")

    ids_in_order = [j["id"] for j in jobs.list_jobs()]
    assert ids_in_order.index(second) < ids_in_order.index(first)

    _wait_until(lambda: jobs.get_job(first)["done"] and jobs.get_job(second)["done"])


def test_stop_job_terminates_a_long_running_process():
    job_id = jobs.start_job(_quick_cmd("import time; time.sleep(30)"), label="test: long sleeper")
    assert jobs.get_job(job_id)["done"] is False

    assert jobs.stop_job(job_id) is True
    _wait_until(lambda: jobs.get_job(job_id)["done"], timeout=5.0)


def test_stop_job_returns_false_for_unknown_id():
    assert jobs.stop_job("does-not-exist") is False


def test_get_report_returns_none_without_a_report_path():
    job_id = jobs.start_job(_quick_cmd("print('hi')"), label="test: no report")
    _wait_until(lambda: jobs.get_job(job_id)["done"])
    assert jobs.get_report(job_id) is None
