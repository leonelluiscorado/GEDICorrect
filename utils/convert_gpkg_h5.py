## Convert GEDI .gpkg to .h5

import tables
import geopandas as gpd
import os
import argparse

parser = argparse.ArgumentParser(description='An auxiliary script to convert GEDI GPKG files into H5.')

parser.add_argument('--files_dir', required=True, help='Path directory to GEDI files that are GPKG.', type=str)
parser.add_argument('--out_dir', required=True, help='Output directory to save H5 files.', type=str)

args = parser.parse_args()

files_dir = args.files_dir
out_dir = args.out_dir

all_files = [f for f in os.listdir(files_dir) if f.endswith(".gpkg")]

groups = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

def init_subgroups(group, file):
    for subgroup in subgroups:
        if subgroup == 'shot_number_x' or subgroup == 'index':
            file.create_earray(group, subgroup, tables.UInt16Atom(), shape=(0,), chunkshape=(1,), expectedrows=1000)
        else:
            file.create_earray(group, subgroup, tables.Float32Atom(), shape=(0,), chunkshape=(1,), expectedrows=1000)

for file in all_files:

    print(f"Processing file {file}")

    file_df = gpd.read_file(os.path.join(files_dir, file), engine='pyogrio')
    subgroups = [col for col in file_df.columns if not 'rh' in col][1:]
    grouped = file_df.groupby('BEAM')

    # Write to file
    with tables.open_file(os.path.join(out_dir, file[:-5]+".h5"), "w") as h5file:

        for beam, group in grouped:

            group_node = h5file.create_group("/", beam)
            
            # For each column in the group, create a dataset in the H5 file
            for col_name in group.columns:
                # Skip the 'BEAM' column since it's used for grouping
                if col_name == 'BEAM' or col_name == 'geometry':
                    continue
                
                # Get the column data
                col_data = group[col_name].values

                if col_data.dtype == object:
                    # Convert string data to fixed-length ASCII
                    col_data = col_data.astype('S')  # Convert to fixed-length string (bytes)
                
                # Create an array in the group for this column
                h5file.create_array(group_node, col_name, col_data, '')

    del file_df, subgroups, grouped