## Convert gpkg to .h5

import tables
import geopandas as gpd
import os

#file = "/home/yoru/Desktop/GEDI01_B_2019156193255_O02717_03_T03903_02_005_01_V002.gpkg"
files_dir = "/home/yoru/3dsmos/RAW_GEDI_DATA/GEDI01_B.V2/FUELSAT"
all_files = [f for f in os.listdir(files_dir) if not 'h5' in f]
out_dir = "/home/yoru/3dsmos/RAW_GEDI_DATA/GEDI01_B.V2/FUELSAT/h5"
#file_df = gpd.read_file(file, engine='pyogrio')

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