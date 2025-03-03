"""
Helper script with data processing functions
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import tables
from scipy.spatial import ConvexHull
import laspy
from shapely.geometry import box, Point, Polygon
from tqdm import tqdm

import os

def create_buffer(footprint, distance):
    """
    Returns Box buffer around **footprint** centroid
    """
    centroid = footprint.geometry.centroid
    return box(centroid.x - distance, centroid.y - distance, centroid.x + distance, centroid.y + distance)


def get_convex_hull(points):
    """
    Builds a bounding box using the Convex Hull algorithm for the ALS point cloud
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return Polygon(hull_points)

def generate_grid(x_max, y_max, step=1):
    '''
    Generates a x_max x y_max grid (NxM) combinations list at 'step' steps.
    Returns an 'offsets' list with all of the possible transformations.
    Each offset is a tuple.
    '''
    offsets = []

    half_xmax = x_max // 2
    half_ymax = y_max // 2

    for x in range(-half_xmax, half_xmax+1, step):
        for y in range(-half_ymax, half_ymax+1, step):
            offsets.append((x, y))

    return offsets


def get_las_extents(las_files_dir, algorithm="convex"):
    """
    Builds extent bounds in every .las file inside **las_files_dir**
    The **algorithm** argument selects the bounding box strategy between 'simple' (bounding box of min and max)
    or 'convex' (uses convex hull to build the extent, may take more time)

    It also checks if a bounding of the ALS already exists as a shapefile, to save time for future processing and
    repeating experiments
    """
    las_extents = {}

    las_files = [f for f in os.listdir(las_files_dir) if (f.endswith('.las') or f.endswith('.laz'))]
    shp_file = [f for f in os.listdir(las_files_dir) if (f.endswith('.shp') and f"CorrectALSBounds" in f)]

    if len(las_files) == 0:
        raise Exception("No LAS files found in specified directory.")
        return

    # Parse CRS and return it
    with laspy.open(os.path.join(las_files_dir, las_files[0])) as las:
        crs = las.header.parse_crs()
        print(f"LAS CRS is {crs}")

    if len(shp_file) != 0:
        # Bounds shapefile found, use it as bounds
        shp_path = os.path.join(las_files_dir, shp_file[0])
        
        print(f"Shapefile found: {shp_path}... Using it as bounds")

        gdf = gpd.read_file(shp_path)
        if 'file_name' not in gdf.columns:
            raise ValueError("The shapefile must contain a file_name column to match LAS file names")
        
        for _, row in gdf.iterrows():
            las_extents[row['file_name']] = row['geometry']

        return las_extents, crs

    print("Processing LAS bounds...")
    with tqdm(total=len(las_files)) as pbar:
        for file in las_files:
            las_file = os.path.join(las_files_dir, file)
            with laspy.open(las_file) as las:

                # Ignore LAS with different CRS
                if las.header.parse_crs() != crs:
                    pbar.update(1)
                    continue

                if algorithm == "simple":
                    # Assume las file CRS is the same as the transformed footprints
                    min_x, min_y, max_x, max_y = las.header.mins[0], las.header.mins[1], las.header.maxs[0], las.header.maxs[1]
                    las_extents[file] = box(min_x, min_y, max_x, max_y)
                    pbar.update(1)

                if algorithm == "convex":
                    las_p = las.read()
                    points = np.vstack((las_p.x, las_p.y)).T
                    # Build bounds using Convex Hull
                    las_extents[file] = get_convex_hull(points)
                    del las_p, points
                    pbar.update(1)

    # Create a GeoDataFrame for the bounds
    gdf = gpd.GeoDataFrame(
        {"file_name": list(las_extents.keys())},
        geometry=list(las_extents.values()),
        crs=crs.to_wkt() if crs else None,
    )

    # Save as a shapefile
    shp_output_path = os.path.join(las_files_dir, f"CorrectALSBounds_{algorithm}.shp")
    gdf.to_file(shp_output_path)
    print(f"ALS Bounds Shapefile saved at: {shp_output_path}")

    return las_extents, crs


def find_intersecting_las_files(footprint_buffer, las_extents):
    """
    Helper function to map to input dataframe.
    Returns a list of all the intersecting las files for each footprint
    """
    intersecting_files = []

    for las_file, extent in las_extents.items():
        # If entire footprint buffer is *within* extent, add it for processing
        if footprint_buffer.within(extent):
            intersecting_files.append(las_file)

    return intersecting_files


def clean_cols_rh(columns, original=False):
    """
    Helper function to parse Relative Height Metrics columns from a dataframe.
    Keeps the selected **rh_col_types** columns

    Original files keep the names like 'rh98' and not original files keep like 'rh_98'
    """
    rh_col_types = [f'{i}' for i in range(25, 105, 5)]
    rh_cols = [col for col in columns if 'rh' in col]

    rh_final_cols = []
    for col in rh_cols:

        # Check only rh number
        if original:
            col_to_check = col.split("rh")[-1]
        else:
            col_to_check = col.split("_")[-1]
        
        # If rh number equal to any on the rh_col_types, add to final rh_col
        if any(col_to_check == rh_col_type for rh_col_type in rh_col_types):
            rh_final_cols.append(col)

    # Exclude original 'rh' columns to include in final value
    columns = [x for x in columns if x not in rh_cols]
    
    return columns + rh_final_cols


def parse_simulated_h5(h5_file, num_sim_points):
    """
    Parses the .h5 file from the GediRat simulation, returning a DataFrame
    """
    data_dict = {}

    with tables.open_file(h5_file, "r") as h5f:
        for node in h5f.list_nodes(where='/'):
            if node.name in ["NBINS", "NPBINS", "Z0", "ZG", "ZGDEM", "ZN", "RXWAVECOUNT", "FSIGMA", "PRES"]:
                if node[:].ndim >= 2:
                    data_dict[node.name] = [x for x in node[:].tolist()]
                else:
                    if len(node) == 1:
                        data_dict[node.name] = num_sim_points * [node[0]]
                    else:
                        data_dict[node.name] = node[:]

    waveform_df = pd.DataFrame.from_dict(data=data_dict)

    return waveform_df


def parse_txt(origin_shotnum, filename):
    """
    Parses the .txt file from the GediMetrics simulation, returning a DataFrame
    """
    with open(filename, "r+") as text_file:
        lines = text_file.readlines()

        # Do things to first line
        first_line = lines[0][2:-2]
        first_line = first_line.strip(' ').split(',')
        first_line = ["_".join(atr.split()) for atr in first_line] # Remove last space
        columns = [column.partition('_') for column in first_line]
        columns = [tuple[2] for tuple in columns]
        first_line = " ".join(columns)
        lines[0] = first_line + '\n'

        text_file.seek(0)
        text_file.writelines(lines)
        text_file.truncate()

    # Rearrange ID into Shot Number and Beam
    df = pd.read_csv(filename, delimiter=' ')
    df['shot_number'] = str(origin_shotnum)
    df['linkM'], df['linkCov'] = 0, 0

    # Rearrange columns
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    cols = clean_cols_rh(cols)

    return df[cols]