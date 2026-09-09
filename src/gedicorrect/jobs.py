"""Persistent background-job management for the local browser interface."""

import json
import os
import re
import signal
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path


JOB_STATE_ENV = "GEDICORRECT_JOB_STATE"
DEFAULT_JOB_STATE = Path.home() / ".gedicorrect" / "job.json"
_PROCESS_HANDLES = {}
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _state_path():
    return Path(os.environ.get(JOB_STATE_ENV, DEFAULT_JOB_STATE)).expanduser().resolve()


@dataclass(frozen=True)
class JobRecord:
    """Serializable state for one UI-managed correction process."""

    pid: int
    command: tuple[str, ...]
    log_path: str
    output_dir: str
    started_at: str
    process_token: str | None = None
    status: str = "running"
    return_code: int | None = None
    finished_at: str | None = None

    @property
    def running(self):
        return self.status == "running"

    def to_dict(self):
        values = asdict(self)
        values["command"] = list(self.command)
        return values

    @classmethod
    def from_dict(cls, values):
        values = values.copy()
        values["command"] = tuple(values["command"])
        return cls(**values)


def _process_snapshot(pid):
    """Return a Linux process start token and state, when available."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except (IndexError, OSError):
        return None, None


def _same_process_is_running(job):
    token, state = _process_snapshot(job.pid)
    if token is not None:
        return state != "Z" and (job.process_token is None or token == job.process_token)

    try:
        os.kill(job.pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _write_job(job):
    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as state_file:
        json.dump(job.to_dict(), state_file, indent=2)
        state_file.write("\n")
    os.replace(temporary_path, state_path)


def _read_job():
    try:
        with open(_state_path(), encoding="utf-8") as state_file:
            return JobRecord.from_dict(json.load(state_file))
    except (KeyError, OSError, TypeError, ValueError):
        return None


def get_job():
    """Load the latest UI job and refresh its process status."""

    job = _read_job()
    if job is None or not job.running:
        return job

    process = _PROCESS_HANDLES.get(job.pid)
    if process is not None:
        return_code = process.poll()
        if return_code is None:
            return job
        _PROCESS_HANDLES.pop(job.pid, None)
        status = "completed" if return_code == 0 else "failed"
        job = replace(job, status=status, return_code=return_code, finished_at=_utc_now())
        _write_job(job)
        return job

    if _same_process_is_running(job):
        return job

    job = replace(job, status="stopped", finished_at=_utc_now())
    _write_job(job)
    return job


def start_job(command, output_dir):
    """Start and persist one correction process."""

    previous_job = get_job()
    if previous_job is not None and previous_job.running:
        raise RuntimeError(f"GEDICorrect is already running as process {previous_job.pid}.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_path / f"gedicorrect-{timestamp}.log"

    process = None
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        process_token, _ = _process_snapshot(process.pid)
        job = JobRecord(
            pid=process.pid,
            command=tuple(command),
            log_path=str(log_path),
            output_dir=str(output_path),
            started_at=_utc_now(),
            process_token=process_token,
        )
        _PROCESS_HANDLES[process.pid] = process
        _write_job(job)
        return job
    except Exception:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise


def cancel_job(job):
    """Terminate a running correction and persist its cancelled state."""

    current_job = get_job()
    if current_job is None or current_job.pid != job.pid:
        return current_job
    if current_job.running:
        try:
            os.killpg(current_job.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        current_job = replace(current_job, status="cancelled", finished_at=_utc_now())
        _write_job(current_job)
    return current_job


def clear_job():
    """Remove the persisted record after a job has stopped."""

    current_job = get_job()
    if current_job is not None and current_job.running:
        raise RuntimeError("A running correction cannot be cleared.")
    try:
        _state_path().unlink()
    except FileNotFoundError:
        pass


def _normalize_terminal_output(output):
    """Render carriage-return progress updates as readable log lines."""

    output = _ANSI_ESCAPE.sub("", output).replace("\r\n", "\n")
    rendered_lines = []
    for line in output.split("\n"):
        if "\r" in line:
            updates = line.split("\r")
            line = next((update for update in reversed(updates) if update), "")
        rendered_lines.append(line.rstrip())
    return "\n".join(rendered_lines)


def read_job_log(job, max_characters=30000):
    """Read the most recent portion of a correction log."""

    try:
        with open(job.log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - max_characters))
            output = log_file.read().decode("utf-8", errors="replace")
            return _normalize_terminal_output(output)
    except OSError:
        return "Waiting for job output..."
