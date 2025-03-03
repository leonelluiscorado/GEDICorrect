import numpy as np
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

our_filespath = "/home/lcorado/correct_GEDI_geoloc/GEDICorrect/output_fuelsat_CCtest"
#theirs_shpfile = "/home/yoru/personal/GEDICorrection/Steven_test90h.shp"
'''
#test_folders = [f for f in os.listdir(our_filespath)]
#test_folders = sorted(test_folders, key=lambda x: int(x.split("_")[0]))

test_results = pd.DataFrame(columns=['criteria', 'r2', 'rmse', 'mae'])
criteria_list = []
r2_list = []
rmse_list = []
mae_list = []

our_filespath = "/home/yoru/output_Fuelsat_1"
theirs_shpfile = "/home/yoru/personal/GEDICorrection/Steven_test90h.shp"
'''
files = [f for f in os.listdir(our_filespath) if f.endswith(".gpkg") and "CORRECTED" in f]

main_df = []

for file in files:
    temp_df = gpd.read_file(os.path.join(our_filespath, file), engine='pyogrio')
    main_df.append(temp_df)

main_df = gpd.GeoDataFrame(pd.concat(main_df))

joined_df = main_df.dropna(axis=0)

print(f"fpts: {len(joined_df['rhGauss_95'].values)}")

r1 = pearsonr(joined_df['rhGauss_95'].values, joined_df['rh95'].values)
r2 = pearsonr(joined_df['rhGauss_95'].values, joined_df['rh95'].values)
rmse = root_mean_squared_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
mae = mean_absolute_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
rsquared_criteria = r1.statistic ** 2

print(f" ---   TEST   ---")
print("Test  R2 : ", r2.statistic ** 2)
print("Test RMSE: ", rmse)
print("Test MAE :", mae)
print("-------------------------")