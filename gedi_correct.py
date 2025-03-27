import os
import argparse

from src.correct import GEDICorrect

# --------------------------COMMAND LINE ARGUMENTS AND ERROR HANDLING---------------------------- #
# Set up argument and error handling
parser = argparse.ArgumentParser(description='A script to correct GEDI Geolocation at the footprint level.')

parser.add_argument('--granules_dir', required=False, help='Local directory where all GEDI files ', type=str)

parser.add_argument('--input_file', required=False, help='GEDI File to be processed and corrected', type=str)

parser.add_argument('--las_dir', required=True, help='Directory of .LAS files required for processing. Must intersect with input granule file(s)', type=str)

parser.add_argument('--out_dir', required=True, help='Directory in which to save the corrected input granules and simulated points', type=str)

parser.add_argument('--save_sim_points', required=False, help='Option to save all the simulated points around each footprint from the input data.',
                    action='store_true', default=False)

parser.add_argument('--save_origin_location', required=False, help='Flag option to save the original location simulated footprint.', action='store_true', default=False)

parser.add_argument('--mode', required=True, help='Selects the footprint correction method between Orbit-level, Beam-level or Footprint-level, based on the list [“orbit”, “beam”, “footprint”].', \
                    type=str, default="footprint")

parser.add_argument('--criteria', required=True, help='Set of criteria to select the best footprint. Select from "wave", "rh", "rh_correlation" and "terrain". \
                                                       Select "all" to evaluate all simulated footprints with all the possible criteria', type=str, default='kl')

parser.add_argument('--grid_size', required=False, help='Specifies the size of the grid for “Orbit-level” or “Beam-level” correction methods.', type=int, default=15)

parser.add_argument('--grid_step', required=False, help='Specifies the step size for the grid for “Orbit-level” or “Beam-level” correction methods.', type=int, default=1)

parser.add_argument('--n_points', required=False, help='Number of points to simulate around each input footprint', type=int, default=100)

parser.add_argument('--radius', required=False, help='Maximum value for radius to simulate points around each original footprint', type=float, default=12.5)

parser.add_argument('--min_dist', required=False, help='Minimum distance between simulated points around each original footprint', type=float, default=1.0)

parser.add_argument('--parallel', required=False, help='Use parallel processing with "--n_processes" processes. If no n_processes are defined, defaults to 4 processes.',
                    action='store_true')

parser.add_argument('--n_processes', required=False, help='Number of processes to use for parallel processing.', type=int, default=4)

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------- #

# List Files and Create Output directory if needed

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
    print(f"Creating output directory {args.out_dir}")

input_granules = None

if args.granules_dir:
    input_granules = [os.path.join(args.granules_dir, f) for f in os.listdir(args.granules_dir) if f.endswith('.gpkg')]

if args.input_file:
    input_granules = [args.input_file]

correct = GEDICorrect(granule_list=input_granules,
                      las_dir=args.las_dir,
                      out_dir=args.out_dir,
                      mode=args.mode,
                      criteria=args.criteria,
                      save_sim_points=args.save_sim_points,
                      save_origin_location=args.save_origin_location,
                      use_parallel=args.parallel,
                      n_processes=args.n_processes)

if args.mode == "footprint":
    results = correct.simulate(n_points=args.n_points, max_radius=args.radius, min_dist=args.min_dist)
else:
    results = correct.simulate(grid_size=args.grid_size, grid_step=args.grid_step)

print(f"[Correction] Correction of input footprints complete! All files have been saved to {args.out_dir}")