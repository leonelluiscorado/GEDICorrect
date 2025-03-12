import numpy as np
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import argparse

parser = argparse.ArgumentParser(description='An auxiliary script to calculate evaluating metrics between corrected and original \
                                              GEDI footprints, comparing RH95.')

parser.add_argument('--result_dir', required=True, help='Path directory to correct GEDI files.', type=str)

args = parser.parse_args()

result_dir = args.result_dir

def rmspe(y_pred, y_true):
    percentage_errors = (y_true - y_pred) / y_true
    return np.sqrt(np.mean(percentage_errors ** 2))

def create_beautiful_scatterplot(df, title, out_file=None):
    # Calculate R-squared, RMSE, RMSPE
    y_true = df['rh95'].values #rh95
    y_pred = df['rhGauss_95'].values #rhGauss_95_x
    r_squared = pearsonr(df['rhGauss_95'].values, df['rh95'].values).statistic**2
    rmse = root_mean_squared_error(y_pred, y_true)
    rmspe_v = rmspe(y_pred, y_true)
    n_points = len(df)
    
    # Set Seaborn theme for aesthetics
    sns.set_style('ticks')
    palette = sns.color_palette("dark")

    # Create the scatterplot
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=df, x='rhGauss_95', y='rh95', s=50, color='#196e27', edgecolor='w', alpha=0.7)

    # Add regression line
    sns.regplot(data=df, x='rhGauss_95', y='rh95', scatter=False, ax=scatter, color='#e71c7d', line_kws={'linewidth': 1.5}, label='Adjusted Linear Regression')

    # Add 1-to-1 line
    plt.plot([0, 30], [0, 30], linestyle='--', color='gray', linewidth=1.5)
    metrics_text = f'$R^2$: {r_squared:.5f}\nRMSE: {rmse:.5f} (m)\nrRMSE {rmspe_v:.5f} (%)\n$N =${n_points}'
    
    # Annotate R-squared value
    plt.text(22, 2, metrics_text, fontsize=13, color='black', bbox=dict(facecolor='white', alpha=1))
    plt.text(28.5, 27.5, "1:1", fontsize=10, color='gray', bbox=dict(facecolor='white', alpha=0))

    # Set limits
    plt.xlim(0, 30)
    plt.ylim(0, 30)

    # Make axes labels bold
    plt.xlabel('Simulated RH95', fontsize=13, fontweight='bold')
    plt.ylabel('Reported RH95', fontsize=13, fontweight='bold')

    plt.title(title, fontsize=15, fontweight='bold')

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.legend(fontsize=11)

    # Save the figure if out_file is provided
    if out_file:
        plt.savefig(out_file, dpi=600, bbox_inches='tight')

    # Show the plot
    plt.show()

files = [f for f in os.listdir(result_dir) if f.endswith(".shp") or f.endswith(".gpkg")]

if not len(files):
    print("Input directory is empty.")
    exit()

main_df = []

for file in files:
    temp_df = gpd.read_file(os.path.join(our_filespath, file), engine='pyogrio')
    main_df.append(temp_df)

main_df = gpd.GeoDataFrame(pd.concat(main_df))
joined_df = main_df.dropna(axis=0)

r = pearsonr(joined_df['rhGauss_95'].values, joined_df['rh95'].values)
rmse = root_mean_squared_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
mae = mean_absolute_error(joined_df['rh95'].values, joined_df['rhGauss_95'].values)
rsquared_criteria = r.statistic ** 2

print(f" ---   TEST   ---")
print(f"N: {len(joined_df['rhGauss_95'].values)}")
print("Test  R2 : ", rsquared_criteria)
print("Test RMSE: ", rmse)
print("Test MAE :", mae)
print("-------------------------")

create_beautiful_scatterplot(joined_df, title="Beam Level - KL + RH Distance - Area 3", out_file=os.path.join(result_dir, "plot.png"))