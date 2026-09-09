"""Command-line interface for GEDICorrect."""

import argparse
import json
import sys

from . import __version__
from .config import CORRECTION_CRITERIA, CORRECTION_MODES, CorrectionConfig
from .runner import run_correction
from .system import inspect_environment


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gedicorrect",
        description="Correct GEDI geolocation using small-footprint ALS data.",
    )
    parser.add_argument("--version", action="version", version=f"GEDICorrect {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run GEDI geolocation correction.")
    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--granules-dir", "--granules_dir", help="Directory of merged GEDI .gpkg files.")
    input_group.add_argument("--input-file", "--input_file", help="One merged GEDI .gpkg file.")
    run_parser.add_argument("--las-dir", "--las_dir", required=True, help="Directory containing ALS .las files.")
    run_parser.add_argument("--out-dir", "--out_dir", required=True, help="Directory for corrected outputs.")
    run_parser.add_argument("--mode", choices=CORRECTION_MODES, default="footprint")
    run_parser.add_argument(
        "--criteria",
        default="kl",
        help=f"Space-separated criteria or 'all'. Choices: {', '.join(CORRECTION_CRITERIA)}.",
    )
    run_parser.add_argument("--grid-size", "--grid_size", type=int, default=15)
    run_parser.add_argument("--grid-step", "--grid_step", type=int, default=1)
    run_parser.add_argument("--als-crs", "--als_crs", default=None, help="ALS EPSG code when LAS metadata has no CRS.")
    run_parser.add_argument("--als-algorithm", "--als_algorithm", choices=("convex", "simple"), default="convex")
    run_parser.add_argument("--time-window", "--time_window", type=float, default=0.04)
    run_parser.add_argument("--parallel", action="store_true", help="Process footprints concurrently.")
    run_parser.add_argument("--n-processes", "--n_processes", type=int, default=4)
    run_parser.add_argument("--random", action="store_true", help="Use random footprint candidates.")
    run_parser.add_argument("--n-points", "--n_points", type=int, default=100)
    run_parser.add_argument("--radius", type=float, default=12.5)
    run_parser.add_argument("--min-dist", "--min_dist", type=float, default=1.0)
    run_parser.add_argument("--save-sim-points", "--save_sim_points", action="store_true")
    run_parser.add_argument("--save-origin-location", "--save_origin_location", action="store_true")

    check_parser = subparsers.add_parser("check", help="Check GEDI Simulator and CPU availability.")
    check_parser.add_argument("--json", action="store_true", help="Print machine-readable output.")

    ui_parser = subparsers.add_parser("ui", help="Start the local browser UI.")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8501)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare GEDI or ALS input data.")
    prepare_subparsers = prepare_parser.add_subparsers(dest="prepare_command", required=True)
    align_parser = prepare_subparsers.add_parser("align", help="Merge and filter L1B/L2A GeoPackages.")
    align_parser.add_argument("--l1b-dir", required=True)
    align_parser.add_argument("--l2a-dir", required=True)
    align_parser.add_argument("--out-dir", required=True)
    las_parser = prepare_subparsers.add_parser("las", help="Convert LAZ files to LAS.")
    las_parser.add_argument("--las-dir", required=True)

    return parser


def config_from_args(args):
    values = vars(args).copy()
    values.pop("command", None)
    return CorrectionConfig(**values)


def print_environment(as_json=False):
    report = inspect_environment()
    values = {
        "ready": report.ready,
        "cpu_count": report.cpu_count,
        "executables": report.executables,
    }
    if as_json:
        print(json.dumps(values, indent=2))
    else:
        print(f"GEDICorrect {__version__}")
        print(f"Available CPUs: {report.cpu_count}")
        for executable, path in report.executables.items():
            print(f"{executable}: {path or 'NOT FOUND'}")
        print("Status: ready" if report.ready else "Status: GEDI Simulator setup required")
    return 0 if report.ready else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            return print_environment(args.json)
        if args.command == "ui":
            from .ui_launcher import launch_ui
            return launch_ui(host=args.host, port=args.port)
        if args.command == "prepare":
            if args.prepare_command == "align":
                from .utilities import align_gedi_products
                outputs = align_gedi_products(args.l1b_dir, args.l2a_dir, args.out_dir)
            else:
                from .utilities import convert_laz_directory
                outputs = convert_laz_directory(args.las_dir)
            print(f"Prepared {len(outputs)} file(s).")
            return 0

        result = run_correction(config_from_args(args))
        print(f"[Correction] Completed in {result.elapsed_seconds:.2f} seconds.")
        if result.output_files:
            print("[Correction] Output files:")
            for filename in result.output_files:
                print(f"  {filename}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"GEDICorrect error: {error}", file=sys.stderr)
        return 2


def ui_main():
    """Dedicated ``gedicorrect-ui`` entry point."""

    return main(["ui", *sys.argv[1:]])
