"""`deskbot schedule <name> "<cron>"` — registers a taught routine with
Windows Task Scheduler. Supports a common subset of 5-field cron syntax
rather than the full grammar; unsupported patterns raise a clear error
instead of silently doing the wrong thing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

_CRON_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$")
_WEEKDAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]

_HELP = (
    "Unsupported schedule '{cron}'. Supported 5-field cron subset:\n"
    "  'M H * * *'    daily at H:M\n"
    "  'M H * * D'    weekly on day D (0=Sun .. 6=Sat) at H:M\n"
    "  '*/N * * * *'  every N minutes\n"
    "  '0 */N * * *'  every N hours"
)


class UnsupportedScheduleError(ValueError):
    pass


def cron_to_schtasks_args(cron: str) -> list[str]:
    m = _CRON_RE.match(cron.strip())
    if not m:
        raise UnsupportedScheduleError(_HELP.format(cron=cron))
    minute, hour, dom, month, dow = m.groups()

    if minute.startswith("*/") and hour == dom == month == dow == "*":
        return ["/sc", "minute", "/mo", minute[2:]]

    if hour.startswith("*/") and minute in ("0", "*") and dom == month == dow == "*":
        return ["/sc", "hourly", "/mo", hour[2:]]

    if dom == month == dow == "*" and minute.isdigit() and hour.isdigit():
        return ["/sc", "daily", "/st", f"{int(hour):02d}:{int(minute):02d}"]

    if dom == month == "*" and dow.isdigit() and minute.isdigit() and hour.isdigit():
        day = _WEEKDAYS[int(dow) % 7]
        return ["/sc", "weekly", "/d", day, "/st", f"{int(hour):02d}:{int(minute):02d}"]

    raise UnsupportedScheduleError(_HELP.format(cron=cron))


def _run_command() -> str:
    deskbot_path = shutil.which("deskbot")
    if deskbot_path:
        return f'"{deskbot_path}"'
    return f'"{sys.executable}" -m deskbot.cli'


def schedule_routine(name: str, cron: str) -> dict:
    sc_args = cron_to_schtasks_args(cron)  # raises UnsupportedScheduleError if unparseable
    task_name = f"deskbot_{name}"
    run_cmd = f"{_run_command()} run {name}"

    cmd = ["schtasks", "/create", "/tn", task_name, "/tr", run_cmd, *sc_args, "/f"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return {
        "ok": proc.returncode == 0,
        "task_name": task_name,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
