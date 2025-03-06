import os
import geopandas as gpd
import numpy as np
import rasterio as rio
from tqdm import tqdm

'''
GEDI Variables to change:
geolocation_elevation_lastbin; geolocation_elevation_bin0; elev_lowestmode; digital_elevation_model 
'''

def fix_vertical_datum(footprint, raster):

    # Get footprint centroid coordinates
    centroid = footprint.geometry.centroid
    coords = [(centroid.x, centroid.y)]

    # Index with raster (will obtain a value)
    datum_offset = list(raster.sample(coords))
    datum_offset = [offset[0] for offset in datum_offset][0]

    # Alter values according to raster value
    footprint["geolocation_elevation_lastbin"] -= datum_offset
    footprint["geolocation_elevation_bin0"] -= datum_offset
    footprint["elev_lowestmode"] -= datum_offset
    footprint["digital_elevation_model"] -= datum_offset

    return footprint

rst_path = "/home/yoru/personal/GEDICorrection/gedicorrect/utils/GeoidPT08.tif"
gedi_dir = "/home/yoru/personal/GEDICorrection/gedicorrect/Area3_VDATUM"

# Read Geoid .TIF
geoid_raster = rio.open(rst_path)

# Read GEDI Directory
granule_list = [os.path.join(gedi_dir, f) for f in os.listdir(gedi_dir) if f.endswith(".gpkg")]

# Read GEDI file and apply 
if granule_list and len(granule_list) >= 1:
    for granule in granule_list:
        granule_df = gpd.read_file(granule, engine='pyogrio').to_crs("EPSG:3763")

        # Check if file is empty
        if len(granule_df) == 0:
            print(f"File {granule} contains no data. Skipping this granule...")
            continue
        
        # Fix vertical datum on each footprint
        granule_df = granule_df.apply(lambda row: fix_vertical_datum(row, geoid_raster), axis=1)

        # Save file again
        gedi_out_filename = 'VDATUM_' + os.path.basename(granule)
        granule_df.to_file(os.path.join(gedi_dir, gedi_out_filename))
        