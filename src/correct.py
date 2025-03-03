"""
Implementation of GEDICorrect class
"""

import os
import tempfile
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import time
from shapely.geometry import Point

from .data_process import get_las_extents, create_buffer, find_intersecting_las_files, generate_grid
from .simulation import process_footprint, init_random_seed, process_all_footprints
from .scorer import CorrectionScorer
from .waveform_processing import plot_waveform_comparison

from tqdm import tqdm
from p_tqdm import p_map

import multiprocessing
from multiprocessing import Manager, Lock
from functools import partial

class GEDICorrect:
    """
    The GEDICorrect ::class:: handles the footprint correction pipeline.
    At new instance creation, it setups the ALS bounding and verifies which GEDI footprints are within
    the bounds, and they also must be valid (non-empty).
    
    If this setup fails, the class fails to simulate and correct footprints, requiring new setup

    At 'simulate', the user can select between sequential or parallel run as an argument. This function
    performs the entire process of correcting the footprints in a sequence:
        1 - Simulates points around original footprint location
        2 - Scores the simulated footprints
        3 - Selects the best scored simulated footprint and saves to files

    Example usage:
    >> correct = GEDICorrect(granule_list='input_granules',
                      las_dir='las_dir',
                      out_dir='out_dir',
                      criteria='all',
                      save_sim_points=True,
                      use_parallel=True,
                      n_processes=24)

    >> results = correct.simulate(args.n_points, args.radius, args.min_dist)
       ...

    """

    def __init__(self, granule_list, las_dir,
                 out_dir=None,
                 mode='footprint',
                 criteria='kl',
                 save_sim_points=False,
                 save_origin_location=False,
                 use_parallel=False,
                 n_processes=None):
        
        self.granule_list = granule_list
        self.las_dir = las_dir
        self.out_dir = out_dir
        self.mode = mode
        self.save_sim_points = save_sim_points
        self.save_origin_location = save_origin_location
        self.use_parallel = use_parallel
        self.n_processes = n_processes
        self.criteria = criteria

        self.gedi_granules = {}

        self.setup_status = self._setup()


    def simulate(self, grid_size=30, grid_step=1, n_points=100, max_radius=12.5, min_dist=1.0):
        """
        This function performs the correction sequence on given GEDI files to class instance

        Args:
            n_points: Number of points to simulate around original footprint
            max_radius: Maximum radius distance to place points
            min_dist: Minimum distance between each simulated point
        """

        if not self.setup_status:
            # Class not properly setup
            raise Exception("[Simulate] Class not properly setup due to error in GEDI Granules or",
                              " opening LAS files. You must create another instance of this class")
            return

        ## Create temporary directory for temp files
        self.temp_dir = tempfile.TemporaryDirectory()
        print(f"[Simulate] Saving files in temporary directory : {self.temp_dir.name}")

        ## Save ALS List from *las_dir* to temporary directory for simulation
        als_list = [os.path.join(self.las_dir, f) for f in os.listdir(self.las_dir) if (f.endswith('.las') or f.endswith('.laz'))]

        with open(os.path.join(self.temp_dir.name, "alsList.txt"), mode="w") as las_txt_file:
            for als_file in als_list:
                las_txt_file.write(f"{als_file}\n")

        if self.use_parallel:
            self.__parallel_simulate(num_points=n_points, max_radius=max_radius, min_dist=min_dist)
        else:
            self.__sequential_simulate(n_points, max_radius, min_dist)

        # Clean Temp Dir
        del self.temp_dir


    def _setup(self):
        """
        Sanity check of GEDI Files and ALS Files

        returns: setup status, if true, the correction can continue
        """

        print("[Setup] Processing list of GEDI orbits and LAS files")

        ## Open LAS Files and check Coordinate System
        try:
            self.las_extents, self.crs = get_las_extents(las_files_dir=self.las_dir)
            assert len(self.las_extents) != 0  # Check if opened something
        except:
            raise Exception("[Setup] Error opening LAS files. Check above exception for more information. Aborting")
            return False

        # Open GEDI footprint files
        if self.granule_list and len(self.granule_list) >= 1:
            for granule in self.granule_list:
                granule_df = gpd.read_file(granule, engine='pyogrio').to_crs(self.crs)

                # Check if file is empty
                if len(granule_df) == 0:
                    print(f"[Setup] File {granule} contains no data. Skipping this granule...")
                    continue

                granule_df['buffer'] = granule_df.apply(create_buffer, distance=25, axis=1) # Create box buffer of footprint in each centroid
                self.gedi_granules[granule] = granule_df
        else:
            # Error defining gedi_granule
            raise Exception("[Setup] No GEDI granules found. Specify another directory where GEDI",
                             " L1B files exist and create another instance of this class")
            return False

        # Check intersecting GEDI Orbits with LAS Extents
        try:
            for filename, granule_df in list(self.gedi_granules.items()):
                granule_df['intersecting_las'] = granule_df['buffer'].apply(find_intersecting_las_files, las_extents=self.las_extents)
                self.gedi_granules[filename] = granule_df[granule_df["intersecting_las"].str.len() != 0]

                if len(self.gedi_granules[filename]) <= 0:
                    print(f"[Setup] No found footprints that intersect with LAS at {filename}")
                    del self.gedi_granules[filename]
                    continue

                print(f"[Setup] Found {len(self.gedi_granules[filename])} footprints that intersect with LAS at {filename}")
        except:
            raise Exception("[Setup] No Intersecting GEDI granules found. Specify another directory where intersecting GEDI",
                             " L1B files exist, or intersecting LAS Files exist and create another instance of this class")
            return False

        return True
            
    def __check_setup_status(self):
        """
        Returns setup status
        """
        return self.setup_status

    def __save_outputs(self, results, filename, offset=None, beam_offset=None):
        """
        Saves outputs from the results argument into (currently) SHP files only
        If save_sim_points is True, it also outputs all of the simulated points to a SHP file
        """

        if offset:
            selected_rows = []  # List to hold the selected rows from each DataFrame

            # Loop over each DataFrame in the results list
            for df in results:
                # Filter the rows that match the best offset
                filtered_df = df[df['grid_offset'] == offset]

                        # Check if any rows match the best offset
                if not filtered_df.empty:
                    # Extract the first matching row as a Series
                    best_row = filtered_df.iloc[0]
                    # Append the row (as a Series) to the list
                    selected_rows.append(best_row)

            # RXWAVECOUNT array to string
            for footprint in selected_rows:
                footprint['RXWAVECOUNT'] = str(footprint['RXWAVECOUNT'])
                footprint['grid_offset'] = str(footprint['grid_offset'])

            out_df = gpd.GeoDataFrame(selected_rows, crs=self.crs).set_geometry('geometry')
            save_out_filename = filename.split('/')[-1].replace('.gpkg', '.shp')
            out_df.to_file(os.path.join(self.out_dir, 'ORBIT_'+save_out_filename))

            return out_df

        if beam_offset:
        # Correction at the BEAM LEVEL
            selected_rows = []  # List to hold the selected rows from each DataFrame

            # Loop over each DataFrame in the results list
            for df in results:
                # Get the BEAM ID for the current footprint
                beam_id = df['BEAM'].iloc[0]  # Assuming each DataFrame has a single BEAM ID
                best_offset = beam_offset.get(beam_id)

                if best_offset is not None:
                    # Filter the rows that match the best offset for this beam
                    filtered_df = df[df['grid_offset'] == best_offset].copy()

                    # Check if any rows match the best offset
                    if not filtered_df.empty:
                        # Extract the first matching row as a Series
                        best_row = filtered_df.iloc[0].copy()
                        # Append the row (as a Series) to the list
                        selected_rows.append(best_row)

            # RXWAVECOUNT array to string, grid_offset to string
            for footprint in selected_rows:
                footprint['RXWAVECOUNT'] = str(footprint['RXWAVECOUNT'])
                footprint['grid_offset'] = str(footprint['grid_offset'])

            # Convert selected rows to a GeoDataFrame for saving as shapefile
            out_df = gpd.GeoDataFrame(selected_rows, crs=self.crs).set_geometry('geometry')
            save_out_filename = filename.split('/')[-1].replace('.gpkg', '.shp')
            out_df.to_file(os.path.join(self.out_dir, 'BEAM_' + save_out_filename))

            return out_df

        # Return each corrected fpt
        final_df = []
        for fpt in results:
            if len(fpt) != 0:
                best_footprint = fpt.sort_values(by=['final_score'], ascending=[False]).head(1).iloc[0]
                final_df.append(best_footprint)

        if len(final_df) == 0:
            print(f"{filename} contains footprints that are not desirable for correction.  Skipping...")
            return

        if self.save_sim_points:
            # Save sim points to SHP file
            save_df = []
            origin_loc = []
            for gpd_df in results:
                if len(gpd_df) != 0:
                    # RXWAVECOUNT array to str for output purposes
                    gpd_df['RXWAVECOUNT'] = gpd_df['RXWAVECOUNT'].astype(str)
                    origin_loc.append(gpd_df.loc[0])
                    save_df.append(gpd_df)

            # Append to to be saved dataframe
            sim_save_df = gpd.GeoDataFrame(pd.concat(save_df))
            sim_save_df.crs = self.crs
            sim_save_df = sim_save_df.drop(columns=['FSIGMA'])
            sim_save_out_filename = filename.split('/')[-1]
            sim_save_df.to_file(os.path.join(self.out_dir, 'SIMPOINTS_'+sim_save_out_filename))

        for footprint in final_df:
            # RXWAVECOUNT array to str for output purposes
            footprint['RXWAVECOUNT'] = str(footprint['RXWAVECOUNT'])

        ## Save corrected footprints to SHP
        if self.out_dir:
            out_df = gpd.GeoDataFrame(final_df, crs=self.crs).set_geometry('geometry')
            out_filename = filename.split('/')[-1]
            out_df = out_df.drop(columns=['FSIGMA'])
            out_df.to_file(os.path.join(self.out_dir, 'CORRECTED_'+out_filename))


    def __sequential_simulate(self, num_points=100, max_radius=12.5, min_dist=1.0):
        """
        Sequential Run of the correction sequence.

        Args:
            n_points: Number of points to simulate around original footprint
            max_radius: Maximum radius distance to place points
            min_dist: Minimum distance between each simulated point
        """
        for filename, footprint_df in self.gedi_granules.items():
            print(f"[Simulate] Correcting granule {filename}")

            scorer = CorrectionScorer(original_df=footprint_df)

            footprints = [row for i, row in footprint_df.iterrows()]
            processed_fpts = []
            results = []

            with tqdm(total=len(footprints), desc ="Processing Footprints") as pbar:
                for fpt in footprints:

                    processed_fpt = process_footprint(fpt, self.temp_dir.name,
                                                      original_df=footprint_df,
                                                      crs=str(self.crs).split(":")[-1],
                                                      num_points=num_points,
                                                      max_radius=max_radius,
                                                      min_dist=min_dist)

                    processed_fpts.append(processed_fpt)
                    pbar.update(1)

            idx = 0
            for fpt in processed_fpts:
                if len(fpt) == 1 and type(fpt[0]) == int:
                    print(f"[Correction] Footprint with shot_number {fpt[0]} presents a vegetation height difference between ALS collection date and GEDI observation.",
                            "Potential vegetation cut. Skipping its correction...")
                    del processed_fpts[idx]
                idx += 1

            with tqdm(total=len(processed_fpts), desc ="Correcting Footprints") as pbar:
                for fpt in processed_fpts:
                    if len(fpt) > 1:
                        correct_fpt = scorer.score(fpt)
                        results.append(correct_fpt)
                    pbar.update(1)

            print(f"[Simulate] Saving corrected granule {filename}")
            self.__save_outputs(results, filename)

    def process_orbit_level(self, footprint, grid, temp_dir, original_df, filename, crs, scorer, score_dict, lock):
    
        idx = multiprocessing.current_process().pid

        # Simulate
        simulated_df = process_all_footprints(footprint, grid, temp_dir, self.las_dir, original_df, crs)

        # Then Score
        scored_df = scorer.score(simulated_df)

        if len(scored_df) == 0:
            return []

        # Return scored simulation
        temp_score_dict = scored_df.set_index('grid_offset')['final_score'].to_dict()

        # Update the global score_dict in a thread-safe way
        with lock:
            for offset, score in temp_score_dict.items():
                if offset in score_dict:
                    score_dict[offset] += score  # Accumulate the score
                else:
                    score_dict[offset] = score  # Initialize if it doesn't exist

        return scored_df

    def process_beam_level(self, footprint, grid, temp_dir, original_df, crs, scorer, score_dict, lock):

        idx = multiprocessing.current_process().pid

        # Simulate
        simulated_df = process_all_footprints(footprint, grid, temp_dir, self.las_dir, original_df, crs)

        # Then Score
        scored_df = scorer.score(simulated_df)

        if len(scored_df) == 0:
            return []

        beam_id = scored_df['BEAM'].values[0]

        # Return scored simulation
        temp_score_dict = scored_df.set_index('grid_offset')['final_score'].to_dict()

        with lock:
            for offset, score in temp_score_dict.items():
                if offset in score_dict[beam_id]:
                    score_dict[beam_id][offset] += score  # Accumulate the score
                else:
                    score_dict[beam_id][offset] = score  # Introduce newly seen score

        return scored_df

    def __parallel_beam_simulate(self):
        '''
        Parallel Run
        '''
        
        with multiprocessing.Pool(self.n_processes, initializer=init_random_seed) as pool:

            manager = Manager()
            lock = manager.Lock()

            for filename, footprint_df in self.gedi_granules.items():
                print(f"[Simulate] Correcting granule {filename}")
                print(f"[Simulate] Criteria: {self.criteria}")
                scorer = CorrectionScorer(original_df=footprint_df, criteria=self.criteria)

                offsets = generate_grid(x_max=31, y_max=31, step=1)

                score_dict = manager.dict()

                for beam_id in ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011", "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]:
                    score_dict[beam_id] = manager.dict()

                partial_func_processing = partial(self.process_beam_level,
                                                  grid=offsets,
                                                  temp_dir=self.temp_dir.name,
                                                  original_df=footprint_df,
                                                  crs=str(self.crs).split(":")[-1],
                                                  scorer=scorer,
                                                  score_dict=score_dict,
                                                  lock=lock)

                footprints = [row for i, row in footprint_df.iterrows()]
                processed_fpts = []

                with tqdm(total=len(footprints), desc="Processing Points") as pbar:
                    for correct_fpt in pool.imap_unordered(partial_func_processing, footprints):
                        processed_fpts.append(correct_fpt)
                        pbar.update(1)

                processed_fpts = [x for x in processed_fpts if not type(x) is list]

                best_beam_offset = {}
                beam_counts = {}

                # Get each BEAM count
                for fpt in processed_fpts:
                    beam_id = fpt['BEAM'].iloc[0]
                    if beam_id in beam_counts:
                        beam_counts[beam_id] += 1
                    else:
                        beam_counts[beam_id] = 1

                for beam_id, offsets in score_dict.items():
                    # After all footprints are processed, calculate the mean score
                    if beam_id in beam_counts:
                        num_footprints = beam_counts[beam_id]
                    else:
                        num_footprints = 0

                    if num_footprints == 0:
                        continue

                    mean_score_dict = {offset: score / num_footprints for offset, score in offsets.items()}

                    # Select the best offset based on the highest mean score
                    best_offset = max(mean_score_dict, key=mean_score_dict.get)

                    print(f"Best offset for {beam_id}: {best_offset} with mean score: {mean_score_dict[best_offset]}")
                    best_beam_offset[beam_id] = best_offset

                print(f"[Simulate] Saving corrected granule {filename}")
                self.__save_outputs(processed_fpts, filename, beam_offset=best_beam_offset)

                del score_dict


    def __parallel_orbit_simulate(self):
        '''
        Parallel Run
        '''
        
        with multiprocessing.Pool(self.n_processes, initializer=init_random_seed) as pool:

            manager = Manager()
            lock = manager.Lock()

            with tqdm(total=len(self.gedi_granules.items()), desc="Files") as pbar1:
                for filename, footprint_df in self.gedi_granules.items():
                    print(f"[Simulate] Correcting granule {filename}")
                    print(f"[Simulate] Criteria: {self.criteria}")
                    scorer = CorrectionScorer(original_df=footprint_df, criteria=self.criteria)

                    offsets = generate_grid(x_max=31, y_max=31, step=1)

                    score_dict = manager.dict()  # Shared dictionary to store {offset: final_score}

                    partial_func_processing = partial(self.process_orbit_level,
                                                    grid=offsets,
                                                    temp_dir=self.temp_dir.name,
                                                    original_df=footprint_df,
                                                    filename=filename,
                                                    crs=str(self.crs).split(":")[-1],
                                                    scorer=scorer,
                                                    score_dict=score_dict,
                                                    lock=lock)

                    footprints = [row for i, row in footprint_df.iterrows()]
                    processed_fpts = []

                    with tqdm(total=len(footprints), desc="Processing Points") as pbar:
                        for correct_fpt in pool.imap_unordered(partial_func_processing, footprints):
                            processed_fpts.append(correct_fpt)
                            pbar.update(1)

                    processed_fpts = [x for x in processed_fpts if not type(x) is list]

                    # After all footprints are processed, calculate the mean score
                    num_footprints = len(processed_fpts)

                    if num_footprints == 0:
                        print(f"{filename} contains invalid footprints for processing. Skipping...")
                        continue

                    mean_score_dict = {offset: score / num_footprints for offset, score in score_dict.items()}

                    # Select the best offset based on the highest mean score
                    best_offset = max(mean_score_dict, key=mean_score_dict.get)

                    print(f"Best offset: {best_offset} with mean score: {mean_score_dict[best_offset]}")

                    # Save files after correcting
                    print(f"[Simulate] Saving corrected granule {filename}")
                    self.__save_outputs(processed_fpts, filename, offset=best_offset)
                    pbar1.update(1)

                    del score_dict

    def __parallel_simulate(self, num_points=100, max_radius=12.5, min_dist=1.0):
        """
        Parallel Run of the correction sequence.
        Defaults to (os.cpu_count() - 2) processes if higher than 6 processes, else defaults to 4

        Args:
            n_points: Number of points to simulate around original footprint
            max_radius: Maximum radius distance to place points
            min_dist: Minimum distance between each simulated point
        """
        
        # Init multiprocessing pool with a random seed for each process
        with multiprocessing.Pool(self.n_processes, initializer=init_random_seed) as pool:

            for filename, footprint_df in self.gedi_granules.items():
                print(f"[Simulate] Correcting granule {filename}")
                print(f"[Simulate] Criteria: {self.criteria}")
                scorer = CorrectionScorer(original_df=footprint_df, criteria=self.criteria)

                # Define partial functions for the pool.imap
                partial_func_processing = partial(process_footprint,
                                                  temp_dir=self.temp_dir.name,
                                                  original_df=footprint_df,
                                                  crs=str(self.crs).split(":")[-1],
                                                  num_points=num_points,
                                                  max_radius=max_radius,
                                                  min_dist=min_dist)

                partial_func_correction = partial(scorer.score)

                footprints = [row for i, row in footprint_df.iterrows()]
                results = []
                processed_fpts = []

                with tqdm(total=len(footprints), desc="Processing Points") as pbar:
                    for correct_fpt in pool.imap_unordered(partial_func_processing, footprints):
                        processed_fpts.append(correct_fpt)
                        pbar.update(1)

                idx = 0
                for fpt in processed_fpts:
                    if len(fpt) == 1 and type(fpt[0]) == int:
                        print(f"[Correction] Footprint with shot_number {fpt[0]} presents a vegetation height difference between ALS collection date and GEDI observation. \
                                Potential vegetation cut. Skipping its correction...")
                        del processed_fpts[idx]
                    idx += 1
                
                with tqdm(total=len(processed_fpts), desc="Correcting Points") as pbar:
                    for result in pool.imap_unordered(partial_func_correction, processed_fpts):
                        results.append(result)
                        pbar.update(1)

                # Save files after correcting
                print(f"[Simulate] Saving corrected granule {filename}")
                self.__save_outputs(results, filename)