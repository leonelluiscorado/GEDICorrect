"""Shared execution service for the Python API, CLI, and UI."""

import time
from dataclasses import dataclass
from pathlib import Path

from .config import CorrectionConfig
from .system import require_environment


@dataclass(frozen=True)
class CorrectionResult:
    """Summary returned after a GEDICorrect run."""

    output_files: tuple[str, ...]
    elapsed_seconds: float


def collect_granules(config: CorrectionConfig):
    """Resolve the selected GEDI inputs in deterministic order."""

    if config.input_file:
        return [config.input_file]
    return sorted(str(path) for path in Path(config.granules_dir).glob("*.gpkg"))


def run_correction(config: CorrectionConfig, check_environment=True):
    """Validate configuration and execute the complete correction pipeline."""

    config = config.normalized().validate()
    if check_environment:
        require_environment()

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing the scientific stack is deliberately delayed. This keeps commands
    # such as ``gedicorrect check`` fast and usable during installation diagnosis.
    from .correct import GEDICorrect

    start = time.monotonic()
    correct = GEDICorrect(
        granule_list=collect_granules(config),
        las_dir=config.las_dir,
        out_dir=config.out_dir,
        mode=config.mode,
        random=config.random,
        criteria=config.criteria,
        save_sim_points=config.save_sim_points,
        save_origin_location=config.save_origin_location,
        als_crs=config.als_crs,
        als_algorithm=config.als_algorithm,
        time_window=config.time_window,
        use_parallel=config.parallel,
        n_processes=config.n_processes,
    )

    if config.random:
        outputs = correct.simulate(
            n_points=config.n_points,
            max_radius=config.radius,
            min_dist=config.min_dist,
        )
    else:
        outputs = correct.simulate(grid_size=config.grid_size, grid_step=config.grid_step)

    return CorrectionResult(
        output_files=tuple(str(path) for path in (outputs or [])),
        elapsed_seconds=time.monotonic() - start,
    )
