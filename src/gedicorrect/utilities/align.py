"""Merge GEDI L1B and L2A GeoPackages for GEDICorrect."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def _granule_key(path):
    parts = path.name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot identify GEDI granule key in filename: {path.name}")
    return parts[2]


def align_gedi_products(l1b_dir, l2a_dir, out_dir):
    """Merge matching L1B/L2A GeoPackages and apply recommended filters."""

    l1b_path = Path(l1b_dir).expanduser().resolve()
    l2a_path = Path(l2a_dir).expanduser().resolve()
    output_path = Path(out_dir).expanduser().resolve()
    if not l1b_path.is_dir() or not l2a_path.is_dir():
        raise ValueError("L1B and L2A inputs must be existing directories.")

    l1b_files = sorted(l1b_path.glob("*.gpkg"))
    l2a_files = {_granule_key(path): path for path in l2a_path.glob("*.gpkg")}
    if not l1b_files:
        raise ValueError(f"No L1B GeoPackages found in {l1b_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []
    columns_to_keep = [
        "shot_number",
        "degrade_flag",
        "quality_flag",
        "elev_lowestmode",
        "digital_elevation_model",
        "num_detectedmodes",
        "solar_elevation",
        "sensitivity",
        "rx_assess_sd_corrected",
        "selected_algorithm",
        "rx_processing_back_threshold",
        "rx_processing_mean",
        "rx_processing_smoothwidth_zcross",
        *[f"rh_{index}" for index in range(1, 101)],
    ]

    for l1b_file in l1b_files:
        key = _granule_key(l1b_file)
        if key not in l2a_files:
            print(f"No matching L2A file found for {l1b_file.name}; skipping")
            continue

        l1b_frame = gpd.read_file(l1b_file, engine="pyogrio")
        l2a_frame = gpd.read_file(l2a_files[key], engine="pyogrio")
        missing_columns = [column for column in columns_to_keep if column not in l2a_frame.columns]
        if missing_columns:
            raise ValueError(f"L2A file {l2a_files[key].name} is missing columns: {', '.join(missing_columns)}")

        merged_frame = pd.merge(
            l1b_frame,
            l2a_frame[columns_to_keep],
            left_on="shot_number_x",
            right_on="shot_number",
        )
        merged_frame = merged_frame.query("degrade_flag == 0 and quality_flag == 1")
        merged_frame = merged_frame.query("sensitivity >= 0.9")
        merged_frame = merged_frame.query("solar_elevation < 0")
        merged_frame = merged_frame.query("rh_95 <= 30")
        merged_frame = merged_frame[~((merged_frame["rh_95"] > 10) & (merged_frame["num_detectedmodes"] == 1))]

        if merged_frame.empty:
            print(f"Filtered merged file {l1b_file.name} is empty; skipping")
            continue

        destination = output_path / l1b_file.name
        gpd.GeoDataFrame(merged_frame, geometry="geometry", crs=l1b_frame.crs).to_file(destination, driver="GPKG")
        saved_files.append(str(destination))
        print(f"Saved {destination}")

    return saved_files
