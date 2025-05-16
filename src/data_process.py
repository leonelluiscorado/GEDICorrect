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
    Returns a box buffer around footprint centroid with a given
    distance, adding to the left, right, top and bottom of footprint
    centroid.

    Args:
        footprint (DataFrame): GEDI Footprint
        distance (int): distance in meters relative to GEDI centroid.

    Returns:
        Box (polygon): Box Polygon boundary coordinates around given footprint. 
    """
    centroid = footprint.geometry.centroid
    return box(centroid.x - distance, centroid.y - distance, centroid.x + distance, centroid.y + distance)


def get_convex_hull(points):
    """
    Builds a bounding box using the Convex Hull algorithm for the ALS point cloud

    Args:
        points (np.array): Set of ALS points.

    Returns:
        Polygon: polygon or boundary of given set of points.
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return Polygon(hull_points)

def generate_grid(x_max, y_max, step=1):
    '''
    Generates a x_max x y_max grid (NxM) combinations list at 'step' steps.
    Returns an 'offsets' list with all of the possible transformations.
    Each offset is a tuple.

    Args:
        x_max (int): Size for x-axis of grid.
        y_max (int): Size for y-axis of grid.
        step (int): Step size between points of grid.

    Returns:
        offsets (list): A list of coordinates for point in grid (offset),
                        where each offset is a tuple.
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
    repeating experiments.

    Args:
        las_files_dir (str): Directory containing .las files.
        algorithm (str): Boundary algorithm.
                         'simple' uses a simple box buffer between x_max and y_max of las points.
                         'convex' uses the Convex Hull algorithm to find a tight-fitting boundary around
                          las points.

    Returns:
        las_extents (dict): A dictionary containing pairs of 'las filename'
                            and a Polygon object describing the boundary.
        crs (pyproj.CRS): Coordinate Reference System parsed from ALS.
                         
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
    Returns a list of all the intersecting las files for each footprint within ALS.

    Args:
        footprint_buffer (Polygon): A box buffer around each footprint to verify
                                    if entire buffer is inside ALS.
        las_extents (dict): A dictionary containing pairs of 'las filename'
                            and a Polygon object describing the boundary.

    Returns:
        intersecting_files (list): List of ALS filenames that intersect with given
                                   footprint buffer.
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
    Keeps the selected **rh_col_types** columns.

    Original files keep the names like 'rh98' and not original files keep like 'rh_98'.

    Args:
        columns (list): A list of column names of a DataFrame.
        original (bool): Flag that changes the 'RH' variable column names.

    Returns:
        list: A list of clean named columns.
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
    Parses the H5 file from the gediRat simulation, returning a DataFrame.

    Args:
        h5_file (str): Filename of the H5 File output by gediRat.
        num_sim_points (int): Number of simulated points by gediRat.

    Returns:
        waveform_df (DataFrame): The H5 file contents parsed into a DataFrame.
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


def parse_txt(origin_footprint, filename):
    """
    Parses the TXT file from the gediMetrics simulation, returning a DataFrame.

    Args:
        origin_shotnum (int): Original shot_number of given footprint to be saved in output DataFrame.
        filename (str): Filename of the CSV file transformed from the TXT file.
    
    Returns:
        df (DataFrame): The TXT file contents parsed into a DataFrame.
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
    df['shot_number'] = str(origin_footprint['shot_number_x'].values[0])
    df['geolocation_delta_time'] = origin_footprint['geolocation_delta_time'].values[0]
    df['linkM'], df['linkCov'] = 0, 0

    # Rearrange columns
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    cols = clean_cols_rh(cols)

    return df[cols]


def cluster_footprints(sim_fpts, method='single', time_window=0.215):
    '''
    Groups footprints into clusters based on the time_window.

    Args:
        processed_fpts (list): List of Dataframes containing the simulated footprints for each input footprint
        time_window (float): The time_window (in Hz) to calculate min and max delta_times to group footprints

    Returns:
        dict of clusters
    '''

    # Create summary table with metadata
    metadata = []
    for idx, df in enumerate(sim_fpts):
        
        # Ignore size one datasets
        if len(df) <= 1:
            continue

        row = df.iloc[0]
        metadata.append({
            'index': idx,
            'geolocation_delta_time': row['geolocation_delta_time'],
            'BEAM': row['BEAM']
        })

    meta_df = pd.DataFrame(metadata).sort_values(by='geolocation_delta_time').reset_index(drop=True)

    clusters = {}

    for i, row in meta_df.iterrows():
        current_idx = row['index']
        current_time = row['geolocation_delta_time']
        current_beam = row['BEAM']

        # Select candidates within ± time_window
        lower = current_time - time_window
        upper = current_time + time_window

        if method == 'single':
            condition = (
                (meta_df['geolocation_delta_time'] >= lower) &
                (meta_df['geolocation_delta_time'] <= upper) &
                (meta_df['BEAM'] == current_beam)
            )
        else:  # 'multi'
            condition = (
                (meta_df['geolocation_delta_time'] >= lower) &
                (meta_df['geolocation_delta_time'] <= upper)
            )

        cluster_indices = meta_df[condition]['index'].tolist()

        # Discard size 1 clusters
        if len(cluster_indices) > 1:
            clusters[current_idx] = cluster_indices

    return dict(sorted(clusters.items()))


def annotate_clusters(processed_fpts, cluster_dict):

    for cluster_id, cluster in cluster_dict.items():

        # Discard size 1 clusters
        if len(cluster) == 1:
            continue

        # Get First and Last footprints of cluster
        first_id, last_id = cluster[0], cluster[-1]

        first_cluster_fpt = processed_fpts[first_id]
        last_cluster_fpt = processed_fpts[last_id]

        # Search (0, 0) on grid, which is the position of the original footprint
        first_original_index = first_cluster_fpt[first_cluster_fpt['grid_offset'] == (0, 0)]
        last_original_index = last_cluster_fpt[last_cluster_fpt['grid_offset'] == (0, 0)]

        # Get latitude and longitude coordinates from the footprint (0, 0) on grid
        lat1, lon1 = first_original_index['lat'].values[0], first_original_index['lon'].values[0]
        lat2, lon2 = last_original_index['lat'].values[0], last_original_index['lon'].values[0]
        cluster_bounds = ((lat1, lon1), (lat2, lon2))

        for fpt in cluster:
            df = processed_fpts[cluster_id].copy()
            df['cluster_id'] = cluster_id
            df['cluster_bounds'] = [cluster_bounds] * len(df)
            processed_fpts[cluster_id] = df

    return processed_fpts


def build_cluster_rectangles(processed_fpts, cluster_dict, crs="EPSG:4326", width_meters=25):
    
    from shapely.geometry import Polygon
    from pyproj import Geod, CRS, Transformer

    input_crs = CRS.from_user_input(crs)
    transformer_to_wgs84 = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
    transformer_to_original = Transformer.from_crs("EPSG:4326", input_crs, always_xy=True)

    geod = Geod(ellps="WGS84")
    cluster_polygons = []

    for cluster_id, indices in cluster_dict.items():

        # Get first/last point in original CRS
        first = processed_fpts[indices[0]].iloc[0]
        last = processed_fpts[indices[-1]].iloc[0]

        # Transform to WGS84
        lon1, lat1 = transformer_to_wgs84.transform(first['lon'], first['lat'])
        lon2, lat2 = transformer_to_wgs84.transform(last['lon'], last['lat'])

        azimuth_fwd, _, _ = geod.inv(lon1, lat1, lon2, lat2)

        offset_angle1 = azimuth_fwd + 90
        offset_angle2 = azimuth_fwd - 90

        l1_lon, l1_lat, _ = geod.fwd(lon1, lat1, offset_angle1, width_meters / 2)
        r1_lon, r1_lat, _ = geod.fwd(lon1, lat1, offset_angle2, width_meters / 2)
        l2_lon, l2_lat, _ = geod.fwd(lon2, lat2, offset_angle1, width_meters / 2)
        r2_lon, r2_lat, _ = geod.fwd(lon2, lat2, offset_angle2, width_meters / 2)

        # Transform back to original CRS
        l1_x, l1_y = transformer_to_original.transform(l1_lon, l1_lat)
        l2_x, l2_y = transformer_to_original.transform(l2_lon, l2_lat)
        r1_x, r1_y = transformer_to_original.transform(r1_lon, r1_lat)
        r2_x, r2_y = transformer_to_original.transform(r2_lon, r2_lat)

        polygon = Polygon([
            (l1_x, l1_y),
            (l2_x, l2_y),
            (r2_x, r2_y),
            (r1_x, r1_y)
        ])

        cluster_polygons.append({
            'cluster_id': cluster_id,
            'num_footprints': len(indices),
            'geometry': polygon
        })

    return gpd.GeoDataFrame(cluster_polygons, crs=crs)
