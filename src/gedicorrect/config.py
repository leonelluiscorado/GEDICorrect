"""Configuration and validation for GEDICorrect runs."""

from dataclasses import asdict, dataclass
from pathlib import Path


CORRECTION_MODES = ("orbit", "beam", "footprint")
CORRECTION_CRITERIA = (
    "wave_pearson",
    "wave_spearman",
    "wave_distance",
    "kl",
    "rh_distance",
    "terrain",
)
DEFAULT_ALL_CRITERIA = tuple(item for item in CORRECTION_CRITERIA if item != "wave_spearman")


@dataclass
class CorrectionConfig:
    """User-facing settings for one GEDICorrect execution."""

    las_dir: str
    out_dir: str
    granules_dir: str | None = None
    input_file: str | None = None
    mode: str = "footprint"
    criteria: str = "kl"
    grid_size: int = 15
    grid_step: int = 1
    als_crs: str | None = None
    als_algorithm: str = "convex"
    time_window: float = 0.04
    parallel: bool = False
    n_processes: int = 4
    random: bool = False
    n_points: int = 100
    radius: float = 12.5
    min_dist: float = 1.0
    save_sim_points: bool = False
    save_origin_location: bool = False

    def validate(self, check_paths=True):
        """Validate settings and return this configuration."""

        if bool(self.granules_dir) == bool(self.input_file):
            raise ValueError("Select exactly one GEDI input: granules_dir or input_file.")

        if self.mode not in CORRECTION_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Select one of: {', '.join(CORRECTION_MODES)}.")

        selected_criteria = list(DEFAULT_ALL_CRITERIA) if self.criteria == "all" else self.criteria.split()
        invalid_criteria = [item for item in selected_criteria if item not in CORRECTION_CRITERIA]
        if not selected_criteria or invalid_criteria:
            invalid = ", ".join(invalid_criteria) if invalid_criteria else "none selected"
            raise ValueError(f"Invalid correction criteria: {invalid}.")

        if "wave_pearson" in selected_criteria and "wave_spearman" in selected_criteria:
            raise ValueError("Select either wave_pearson or wave_spearman, not both.")

        if self.als_algorithm not in ("convex", "simple"):
            raise ValueError("ALS boundary algorithm must be 'convex' or 'simple'.")

        if self.grid_size < 1 or self.grid_step < 1:
            raise ValueError("Grid size and grid step must be positive integers.")
        if self.time_window <= 0:
            raise ValueError("Time window must be greater than zero.")
        if self.n_processes < 1:
            raise ValueError("Number of processes must be at least one.")
        if self.n_points < 1 or self.radius <= 0 or self.min_dist <= 0:
            raise ValueError("Random point count, radius, and minimum distance must be positive.")
        if self.random and self.mode != "footprint":
            raise ValueError("Random point correction is available only in footprint mode.")

        if check_paths:
            self._validate_paths()

        return self

    def _validate_paths(self):
        las_path = Path(self.las_dir).expanduser()
        if not las_path.is_dir():
            raise ValueError(f"ALS directory does not exist: {las_path}")
        if not any(las_path.glob("*.las")):
            raise ValueError(f"ALS directory contains no .las files: {las_path}")

        if self.granules_dir:
            granules_path = Path(self.granules_dir).expanduser()
            if not granules_path.is_dir():
                raise ValueError(f"GEDI directory does not exist: {granules_path}")
            if not any(granules_path.glob("*.gpkg")):
                raise ValueError(f"GEDI directory contains no .gpkg files: {granules_path}")

        if self.input_file:
            input_path = Path(self.input_file).expanduser()
            if not input_path.is_file() or input_path.suffix.lower() != ".gpkg":
                raise ValueError(f"GEDI input must be an existing .gpkg file: {input_path}")

        output_path = Path(self.out_dir).expanduser()
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path is not a directory: {output_path}")

    def normalized(self):
        """Return a copy whose filesystem paths are expanded and absolute."""

        values = asdict(self)
        for name in ("las_dir", "out_dir", "granules_dir", "input_file"):
            if values[name]:
                values[name] = str(Path(values[name]).expanduser().resolve())
        return CorrectionConfig(**values)

    def to_cli_args(self):
        """Serialize the configuration for a background CLI process."""

        args = ["run", "--las-dir", self.las_dir, "--out-dir", self.out_dir]
        if self.granules_dir:
            args.extend(["--granules-dir", self.granules_dir])
        if self.input_file:
            args.extend(["--input-file", self.input_file])

        args.extend([
            "--mode", self.mode,
            "--criteria", self.criteria,
            "--grid-size", str(self.grid_size),
            "--grid-step", str(self.grid_step),
            "--als-algorithm", self.als_algorithm,
            "--time-window", str(self.time_window),
            "--n-processes", str(self.n_processes),
            "--n-points", str(self.n_points),
            "--radius", str(self.radius),
            "--min-dist", str(self.min_dist),
        ])

        if self.als_crs:
            args.extend(["--als-crs", self.als_crs])
        for enabled, flag in (
            (self.parallel, "--parallel"),
            (self.random, "--random"),
            (self.save_sim_points, "--save-sim-points"),
            (self.save_origin_location, "--save-origin-location"),
        ):
            if enabled:
                args.append(flag)
        return args
