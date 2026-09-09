"""System checks used by the CLI, UI, and container."""

import os
import shutil
from dataclasses import dataclass


REQUIRED_EXECUTABLES = ("gediRat", "gediMetric")


@dataclass(frozen=True)
class EnvironmentReport:
    """Availability report for GEDICorrect's external tools."""

    cpu_count: int
    executables: dict[str, str | None]

    @property
    def ready(self):
        return all(self.executables.values())

    @property
    def missing(self):
        return [name for name, path in self.executables.items() if path is None]


def _cgroup_cpu_limit():
    """Read a Docker/Linux cgroup v2 CPU quota when one is configured."""

    try:
        quota, period = open("/sys/fs/cgroup/cpu.max", encoding="utf-8").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def available_cpu_count():
    """Return CPUs available to this process, including container limits."""

    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))
    cgroup_limit = _cgroup_cpu_limit()
    if cgroup_limit:
        counts.append(cgroup_limit)
    return max(1, min(counts))


def inspect_environment():
    """Inspect external executable and CPU availability."""

    return EnvironmentReport(
        cpu_count=available_cpu_count(),
        executables={name: shutil.which(name) for name in REQUIRED_EXECUTABLES},
    )


def require_environment():
    """Raise a helpful error when GEDI Simulator is unavailable."""

    report = inspect_environment()
    if not report.ready:
        missing = ", ".join(report.missing)
        raise RuntimeError(
            f"Missing GEDI Simulator executable(s): {missing}. "
            "Install GEDI Simulator or use the GEDICorrect Docker image."
        )
    return report
