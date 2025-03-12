"""
Handles the simulation of points around footprints and contains functions for processing the C program of gediSimulator
"""
import os

import multiprocessing
import numpy as np
import geopandas as gpd
import pandas as pd

import laspy
from shapely.geometry import box, Point

import subprocess
from .data_process import parse_txt, parse_simulated_h5

def init_random_seed():
    """
    Initializes a random seed for each multiprocessing process. This works to ensure
    that no other worker process shares the inherited seed from the parent process.
    """
    seed = multiprocessing.current_process().pid  # Use process ID as the seed
    np.random.seed(seed)


def generate_random_points(centroid_x, centroid_y, num_points, max_radius=12.5, min_dist=1.0):
    """
    Generates a random number of 'num_points' points around (x,y) coordinates of a footprint
     up to a 'max_radius' distance, at 'min_dist' intervals between generated points.

    Args:
        centroid_x (float): X coordinate of footprint's centroid.
        centroid_y (float): Y coordinate of footprint's centroid.
        num_points (int): Number of points to simulate around original footprint.
        max_radius (float): Maximum radius distance to place points.
        min_dist (float): Minimum distance between each simulated point.

    Returns:
        points (list): A list of Point objects (with x,y coordinates) which represent the points
                       around each footprint's centroid.
    """
    centroid = Point(centroid_x, centroid_y)
    
    # Define the boundary of the circle within which points will be placed
    boundary = centroid.buffer(max_radius)
    points = []
    
    # Keep trying until all simulated points are valid
    while len(points) < num_points:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, max_radius)
        x = centroid_x + np.cos(angle) * distance
        y = centroid_y + np.sin(angle) * distance
        new_point = Point(x, y)
        
        # Check if the new point is at least min_dist meters away from all other points
        if all(new_point.distance(other) >= min_dist for other in points):
            points.append(new_point)
        
        # If the points list is filled and all are valid, exit the loop
        if len(points) == num_points:
            break

    return points

def process_all_footprints(footprint, temp_dir, las_dir, original_df, crs,
                           grid=None,
                           num_points=None,
                           max_radius=12.5,
                           min_dist=1.0,
                           simulate_original=False):
    '''
    Core of GEDI Simulation footprints (Simulation Unit).
    1 - Generates points around footprint centroid (Random or Grid);
    2 - Runs the desired simulations from GediRat and GediMetrics from GEDI Simulator;
    3 - Parses and processes the output of the simulations to be returned;
    4 - Applies a filtering step to ensure correct simulations pass to the Scoring Unit.

    The final output is a DataFrame with all simulated footprints around the input 'footprint'.
    The output of the GEDI Simulator programs are supressed.

    Args:
        footprint (DataFrame): Single original footprint to simulate and additional info.
        temp_dir (TemporaryDirectory): A temporary directory to keep information
                                       and I/O operations during simulation
        las_dir (str): Directory where .las files are located.
        original_df (DataFrame): The original DataFrame of input footprint GEDI orbit.
        crs (pyproj.CRS): Coordinate Reference System to be used in simulation.
        grid (list): A list of all possible offsets of the given grid (Valid for Orbit-Level or Beam-Level).
        num_points (int): Number of points to simulate around original footprint.
        max_radius (float): Maximum radius distance to place points.
        min_dist (float): Minimum distance between each simulated point.
        simulate_original (bool): Flag to simulate footprint at original position or not.
    '''

    idx = multiprocessing.current_process().pid   # Get current process unique id

    # Shot number
    shot_number = footprint['shot_number_x']
    original_fpt = original_df.loc[original_df['shot_number_x'] == shot_number]

    # Write intersecting las files for each footprint
    with open(os.path.join(temp_dir, f"laslist_{idx}.txt"), "w") as f:
        for las in footprint['intersecting_las']:
            f.write(f"{os.path.join(las_dir, las)}\n")
    # Nbins
    nbins = str(original_fpt['rx_sample_count'].values[0]+1)
    
    ## Generate txt list of coordinates from a grid
    if grid:
        with open(os.path.join(temp_dir, f"points_test_{idx}.txt"), "w") as f:
            if simulate_original:
                f.write(f"{footprint['geometry'].x} {footprint['geometry'].y}\n") # Write original footprint position as first point
                num_points += 1  # Additional point
            for offset in grid:
                offset_x = footprint['geometry'].x + offset[0]
                offset_y = footprint['geometry'].y + offset[1]
                f.write(f"{offset_x} {offset_y}\n")

    ## Generate txt list of coordinates from random points
    if num_points:
        ## Generate random points around footprint
        rand_points = generate_random_points(footprint['geometry'].x, footprint['geometry'].y, num_points=num_points, max_radius=max_radius, min_dist=min_dist)

        with open(os.path.join(temp_dir, f"points_test_{idx}.txt"), "w") as f:
            if simulate_original:
                f.write(f"{footprint['geometry'].x} {footprint['geometry'].y}\n") # Write original footprint position as first point
                num_points += 1  # Additional point
            for point in rand_points:
                f.write(f"{point.x} {point.y}\n")

    h5_file_dir = os.path.join(temp_dir, f"simu_wavef_{idx}.h5")
    points_file_dir = os.path.join(temp_dir, f"points_test_{idx}.txt")
    las_points_dir = os.path.join(temp_dir, f"laslist_{idx}.txt")
    metric_outroot = os.path.join(temp_dir, f"{idx}_")

    ## Simulate waveforms
    exit_code = subprocess.run(["gediRat", "-inList", las_points_dir, "-listCoord", points_file_dir, "-hdf", "-aEPSG", "3763", "-ground", "-maxBins", nbins, "-output", h5_file_dir], stdout=subprocess.DEVNULL)
    exit_code = subprocess.run(["gediMetric", "-input", h5_file_dir, "-readHDFgedi", "-ground", "-varScale", "3.5", "-sWidth", "0.8", "-rhRes", "1", "-laiRes", "5", "-outRoot", metric_outroot], stdout=subprocess.DEVNULL)
    
    ## Handle each output
    txt_df = parse_txt(footprint['shot_number_x'], metric_outroot+'.metric.txt') ######## TODO: Transform shotnumber to string and csv must display differently
    try:
        h5_df  = parse_simulated_h5(h5_file_dir, len(grid))
    except ValueError as e:
        return []

    # Concat the TXT and H5 dataframes
    all_df = pd.concat([txt_df, h5_df], axis=1)

    # Filter out NaN and add Geometry
    #all_df = all_df.dropna(axis=0)
    all_df['geometry'] = list(zip(all_df.lon, all_df.lat))
    all_df['geometry'] = all_df['geometry'].apply(Point)

    if grid:
        all_df['grid_offset'] = grid

    # Filter out special case footprints
    if grid and simulate_original:
        grid_size = len(grid) + 1

    if grid and len(all_df) < grid_size:
        # Did not simulate all points, discard
        return []
    
    if num_points and len(all_df) < num_points:
        # Did not simulate all points, discard
        return []

    # Sanity check: Check if vegetation was cut with original rh95
    original_rh95 = original_fpt['rh_95'].values[0]
    rh95_simulated_position = all_df['rhGauss_95']

    # If mean difference between RH95 of Simulated and GEDI
    mean_diffrh95 = (rh95_simulated_position - original_rh95).mean()
    if mean_diffrh95 < -10:
        # If negative, possibly a vegetation cut and datum difference between ALS and GEDI
        return [shot_number]

    point_df = gpd.GeoDataFrame(all_df, geometry='geometry')

    ## Return corrected footprint 
    return point_df