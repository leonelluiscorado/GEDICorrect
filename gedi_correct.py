import os
import argparse
import pandas as pd
import geopandas as gpd
import scipy
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.correct import GEDICorrect
from src.waveform_processing import create_beautiful_scatterplot
from src.metric import *

import time

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

parser.add_argument('--random', required=False, help='Use random footprint-level correction without clusterization. Used only if mode == \'footprint\'', \
                    action='store_true', default=False)

parser.add_argument('--criteria', required=True, help='Set of criteria to select the best footprint. Select from "wave", "rh", "rh_correlation" and "terrain". \
                                                       Select "all" to evaluate all simulated footprints with all the possible criteria', type=str, default='kl')

parser.add_argument('--grid_size', required=False, help='Specifies the size of the grid for “Orbit-level” or “Beam-level” correction methods.', type=int, default=15)

parser.add_argument('--grid_step', required=False, help='Specifies the step size for the grid for “Orbit-level” or “Beam-level” correction methods.', type=int, default=1)

parser.add_argument('--als_crs', required=False, help='(Optional) Set the EPSG code of ALS if it does not have.', type=str, default=None)

parser.add_argument('--als_algorithm', required=False, help='(Optional) Set the ALS bounding algorithm. Default is convex, which builds tight-fitting boundary. \'simple\' creates a simple bounding box around the ALS.', type=str, default='convex')

parser.add_argument('--time_window', required=False, help='Minimum distance between simulated points around each original footprint', type=float, default=0.04)

parser.add_argument('--parallel', required=False, help='Use parallel processing with "--n_processes" processes. If no n_processes are defined, defaults to 4 processes.',
                    action='store_true')

parser.add_argument('--n_processes', required=False, help='Number of processes to use for parallel processing.', type=int, default=4)

args = parser.parse_args()

# ----------------------------------------------------------------------------------------------- #

# List Files and Create Output directory if needed

start = time.time()

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
                      random=args.random,
                      criteria=args.criteria,
                      save_sim_points=args.save_sim_points,
                      save_origin_location=args.save_origin_location,
                      als_crs=args.als_crs,
                      als_algorithm=args.als_algorithm,
                      use_parallel=args.parallel,
                      n_processes=args.n_processes)

if args.random and args.mode != "footprint":
    args.random = False

if args.random:
    results = correct.simulate(n_points=args.n_points, max_radius=args.radius, min_dist=args.min_dist)
else:
    results = correct.simulate(grid_size=args.grid_size, grid_step=args.grid_step)

print(f"[Correction] Correction of input footprints complete! All files have been saved to {args.out_dir}")

end = time.time()

print(f"Time elapsed for this experiment: {end - start} seconds")

# ----------- Calculate results (RH95 reported vs RH95 simulated) ---------------- #
'''
print(f"[Result] Calculating results from {args.out_dir}")

files = [f for f in os.listdir(args.out_dir) if (f.endswith(".shp") or f.endswith(".gpkg")) and "CORRECTED" in f]

if not len(files):
    print("Output directory is empty.")
    exit()

main_df = []

for file in files:
    temp_df = gpd.read_file(os.path.join(args.out_dir, file), engine='pyogrio')
    main_df.append(temp_df)

main_df = gpd.GeoDataFrame(pd.concat(main_df))
joined_df = main_df.dropna(axis=0)

r = scipy.stats.pearsonr(joined_df['rhGauss_95'].values, joined_df['rh95'].values)
rmse = root_mean_squared_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
mae = mean_absolute_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
rsquared_criteria = r.statistic ** 2

print(f" ---   RESULTS  ---")
print(f"N points: {len(joined_df['rhGauss_95'].values)}")
print("Test  R2 : ", rsquared_criteria)
print("Test RMSE: ", rmse)
print("Test MAE :", mae)
print("-------------------------")

create_beautiful_scatterplot(joined_df, title=f"Results at {args.mode} using {args.criteria}", out_file=os.path.join(args.out_dir, "result_plot.png"))'''