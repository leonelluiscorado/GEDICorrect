import geopandas as gpd
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='An auxiliary script to merge separate GEDI L1B and L2A data products. \
                                              This operation is required for GEDICorrect to work. It also applies \
                                              additional filtering for GEDI footprints.')

parser.add_argument('--l1b_dir', required=True, help='Path directory of L1B GEDI files', type=str)
parser.add_argument('--l2a_dir', required=True, help='Path directory of L2A GEDI files', type=str)
parser.add_argument('--out_dir', required=True, help='Output directory to put merged GEDI files.', type=str)

args = parser.parse_args()

l1b_dir = args.l1b_dir
l2a_dir = args.l2a_dir
out_dir = args.out_dir

l1b_files = [f for f in os.listdir(l1b_dir) if f.endswith(".gpkg")]
l2a_files = [f for f in os.listdir(l2a_dir) if f.endswith(".gpkg")]

l2a_dict = {}

for l1b in l1b_files:
    l1b_ext = l1b.split("_")[2]

    for l2a in l2a_files:
        filename_ext = l2a.split("_")[2]

        if filename_ext == l1b_ext:
            l2a_dict[l1b] = l2a
            break

for file in l1b_files:
    l1b_file = gpd.read_file(os.path.join(l1b_dir, file), engine='pyogrio')
    l2a_file = gpd.read_file(os.path.join(l2a_dir, l2a_dict[file]), engine='pyogrio')

    cols_to_keep = ['shot_number', 'degrade_flag', 'quality_flag', 'elev_lowestmode', 'digital_elevation_model', 
                    'num_detectedmodes', 'solar_elevation', 'sensitivity']
    cols_to_keep = cols_to_keep + [f"rh_{i}" for i in range(1, 101)]

    rename_col = {}
    for i in range(1,101):
        rename_col[f'rh_{i}'] = f'rh_{i}'
    rename_col['shot_number_x'] = 'shot_number'

    l2a_file_to_merge = l2a_file[cols_to_keep]
    l2a_file_to_merge = l2a_file_to_merge.rename(columns=rename_col)

    merged_df = pd.merge(l1b_file, l2a_file_to_merge, left_on='shot_number_x', right_on=[sn for sn in l2a_file_to_merge.columns if sn.endswith('shot_number')][0])

    ### GEDI recommended filtering conditions

    merged_df = merged_df.query('degrade_flag == 0 and quality_flag == 1')

    merged_df = merged_df.query('sensitivity >= 0.9')

    merged_df = merged_df.query('solar_elevation < 0')

    ## GEDICorrect recommended filters

    merged_df = merged_df.query('rh_95 <= 30')

    merged_df = merged_df[~((merged_df['rh_95'] > 10) & (merged_df['num_detectedmodes'] == 1))]

    if len(merged_df) < 1:
        print(f"Filtered merged df {file} is empty, skipping")
        continue

    print("Saving ", file)
    merged_df.to_file(os.path.join(out_dir, file), driver="GPKG")

    del merged_df, l1b_file, l2a_file