"""Local browser interface for GEDICorrect."""

import os
import sys
from pathlib import Path

import streamlit as st

from gedicorrect import __version__
from gedicorrect.config import CORRECTION_CRITERIA, CORRECTION_MODES, CorrectionConfig
from gedicorrect.jobs import cancel_job, clear_job, get_job, read_job_log, start_job
from gedicorrect.system import inspect_environment


st.set_page_config(page_title="GEDICorrect", page_icon="🌍", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        width: clamp(38rem, 34vw, 46rem);
        max-width: calc(100vw - 2rem);
        margin-inline: auto;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stMainBlockContainer"] h1 {
        font-size: clamp(1.8rem, 2vw, 2.2rem);
        line-height: 1.2;
    }

    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3 {
        font-size: clamp(1.15rem, 1.25vw, 1.35rem);
        line-height: 1.3;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    @media (max-width: 42rem) {
        [data-testid="stMainBlockContainer"] {
            width: calc(100vw - 1rem);
            max-width: none;
            padding-inline: 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CONTAINER_MODE = os.environ.get("GEDICORRECT_CONTAINER") == "1"
CONTAINER_ALS_DIR = Path("/data/als")
CONTAINER_INPUT_DIR = Path("/data/input")
CONTAINER_OUTPUT_DIR = Path("/data/output")


def _data_files(directory, suffix):
    """Return supported files from one mounted data directory."""

    try:
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == suffix
        )
    except OSError:
        return []


def _start_job(config):
    command = [sys.executable, "-m", "gedicorrect", *config.to_cli_args()]
    return start_job(command, config.out_dir)


st.title("GEDICorrect")
st.caption(f"GEDI geolocation correction · version {__version__} · processing stays on this computer")

environment = inspect_environment()
ready_message = "GEDI Simulator is ready." if environment.ready else f"Missing: {', '.join(environment.missing)}"
status_col, cpu_col = st.columns(2)
status_col.metric("Environment", "Ready" if environment.ready else "Setup required", ready_message)
cpu_col.metric("Available CPUs", environment.cpu_count, "Container-aware detection")

if not environment.ready:
    st.warning("Install GEDI Simulator or run the official GEDICorrect container before starting a correction.")

job = get_job()
job_running = bool(job and job.running)

with st.container():
    st.subheader("Input and output")
    data_ready = True
    if CONTAINER_MODE:
        las_files = _data_files(CONTAINER_ALS_DIR, ".las")
        gedi_files = _data_files(CONTAINER_INPUT_DIR, ".gpkg")
        output_ready = CONTAINER_OUTPUT_DIR.is_dir() and os.access(CONTAINER_OUTPUT_DIR, os.W_OK)

        als_col, gedi_col, output_col = st.columns(3)
        als_col.metric("ALS files", len(las_files), "Mounted data folder")
        gedi_col.metric("GEDI granules", len(gedi_files), "Mounted data folder")
        output_col.metric("Output folder", "Ready" if output_ready else "Unavailable", "Mounted data folder")

        if gedi_files:
            all_granules = f"All detected granules ({len(gedi_files)})"
            selected_gedi = st.selectbox(
                "GEDI input",
                [all_granules, *(path.name for path in gedi_files)],
                help="Choose one merged GeoPackage or process every detected granule.",
            )
            granules_dir = str(CONTAINER_INPUT_DIR) if selected_gedi == all_granules else None
            input_file = None if granules_dir else str(CONTAINER_INPUT_DIR / selected_gedi)
        else:
            st.selectbox("GEDI input", ["No GeoPackages detected"], disabled=True)
            granules_dir = str(CONTAINER_INPUT_DIR)
            input_file = None

        las_dir = str(CONTAINER_ALS_DIR)
        out_dir = str(CONTAINER_OUTPUT_DIR)
        data_ready = bool(las_files and gedi_files and output_ready)
        if not data_ready:
            st.warning(
                "GEDICorrect cannot start until the selected ALS folder contains .las files, "
                "the GEDI folder contains .gpkg files, and the output folder is writable. "
                "Restart GEDICorrect with different mounted folders to change them."
            )
        else:
            st.caption("Docker is using the data folders selected before launch.")
    else:
        input_kind = st.radio(
            "GEDI input",
            ("Directory of merged files", "Single merged file"),
            horizontal=True,
        )
        input_path = st.text_input(
            "Merged GEDI path",
            value="/path/to/merged-gedi" if input_kind.startswith("Directory") else "/path/to/granule.gpkg",
        )
        las_dir = st.text_input("ALS directory", value="/path/to/als")
        out_dir = st.text_input("Output directory", value="/path/to/results")
        granules_dir = input_path if input_kind.startswith("Directory") else None
        input_file = input_path if input_kind.startswith("Single") else None

    st.subheader("Correction settings")
    mode_col, algorithm_col = st.columns(2)
    mode = mode_col.selectbox("Correction mode", CORRECTION_MODES, index=2)
    als_algorithm = algorithm_col.selectbox("ALS boundary", ("convex", "simple"))
    criteria_list = st.multiselect("Similarity criteria", CORRECTION_CRITERIA, default=["kl"])
    als_crs = st.text_input("ALS EPSG code (optional)", placeholder="For example: 32629")

    if mode != "footprint":
        st.session_state["random_mode"] = False
    random_mode = st.checkbox(
        "Use random candidate points",
        key="random_mode",
        disabled=mode != "footprint",
    )

    if random_mode:
        random_col, radius_col, distance_col = st.columns(3)
        n_points = random_col.number_input("Random points", min_value=1, value=100)
        radius = radius_col.number_input("Maximum radius (m)", min_value=0.1, value=12.5)
        min_dist = distance_col.number_input("Minimum distance (m)", min_value=0.1, value=1.0)
        grid_size = 15
        grid_step = 1
        time_window = 0.04
    else:
        grid_col, step_col, window_col = st.columns(3)
        grid_size = grid_col.number_input("Grid size", min_value=1, value=15, step=2)
        grid_step = step_col.number_input("Grid step (m)", min_value=1, value=1)
        time_window = window_col.number_input(
            "Time window",
            min_value=0.001,
            value=0.04,
            format="%.3f",
        )
        n_points = 100
        radius = 12.5
        min_dist = 1.0

    st.subheader("Performance and outputs")
    parallel = st.checkbox("Parallel processing", value=environment.cpu_count > 1)
    default_processes = min(4, environment.cpu_count)
    n_processes = st.slider(
        "Worker processes",
        min_value=1,
        max_value=environment.cpu_count,
        value=default_processes,
        disabled=not parallel,
    )
    output_col, origin_col = st.columns(2)
    save_sim_points = output_col.checkbox("Save simulated candidate points")
    save_origin_location = origin_col.checkbox("Save original simulated locations")

    submitted = st.button(
        "Run correction",
        type="primary",
        disabled=job_running or not environment.ready or not data_ready,
        use_container_width=True,
    )

if submitted:
    config = CorrectionConfig(
        las_dir=las_dir,
        out_dir=out_dir,
        granules_dir=granules_dir,
        input_file=input_file,
        mode=mode,
        criteria=" ".join(criteria_list),
        grid_size=int(grid_size),
        grid_step=int(grid_step),
        als_crs=als_crs or None,
        als_algorithm=als_algorithm,
        time_window=float(time_window),
        parallel=parallel,
        n_processes=int(n_processes),
        random=random_mode,
        n_points=int(n_points),
        radius=float(radius),
        min_dist=float(min_dist),
        save_sim_points=save_sim_points,
        save_origin_location=save_origin_location,
    )
    try:
        config.validate()
        _start_job(config.normalized())
        st.rerun()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))


@st.fragment(run_every=2 if job_running else None)
def job_monitor():
    current_job = get_job()
    if not current_job:
        return

    st.subheader("Current job")
    if current_job.running:
        st.info(f"Correction is running (process {current_job.pid}).")
        st.caption(f"Started: {current_job.started_at}")
        if st.button("Cancel correction"):
            cancel_job(current_job)
            st.warning("Correction cancelled. Partial outputs and the job log were retained.")
            st.rerun()
    elif current_job.status == "completed":
        st.success("Correction completed successfully.")
        output_files = sorted(
            path for path in Path(current_job.output_dir).glob("*.gpkg")
            if path.name.startswith(("ORBIT_", "BEAM_", "FOOTPRINT_"))
        )
        if output_files:
            st.write("Corrected outputs:")
            for output_file in output_files:
                st.code(str(output_file), language="text", wrap_lines=True)
    elif current_job.status == "cancelled":
        st.warning("Correction was cancelled. Partial outputs and the job log were retained.")
    elif current_job.status == "failed":
        st.error(f"Correction stopped with exit code {current_job.return_code}. Review the log below.")
    else:
        st.warning("The previous correction is no longer running. Review its log for the last recorded output.")

    st.code(read_job_log(current_job), language="text", wrap_lines=True, height=320)
    st.caption(f"Full log: {current_job.log_path}")

    if not current_job.running and st.button("Clear completed job"):
        clear_job()
        st.rerun()


job_monitor()
