"""
Implementation of GEDICorrect class
"""

import os
import tempfile
import geopandas as gpd
import pandas as pd

from .data_process import *
from .simulation import process_all_footprints, init_random_seed, process_all_footprints
from .scorer import CorrectionScorer

from tqdm import tqdm

import multiprocessing
from multiprocessing import Manager
from functools import partial
import gc


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
                      criteria='kl',
                      save_sim_points=True,
                      use_parallel=True,
                      n_processes=24)

    >> results = correct.simulate(args.n_points, args.radius, args.min_dist)
       ...

    """
    _shared_results = None

    def __init__(self, granule_list, las_dir,
                 out_dir=None,
                 mode='footprint',
                 random=False,
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
        self.random = random

        self.gedi_granules = {}

        # Perform setup check: ALS bounding and GEDI
        self.setup_status = self._setup()

    @staticmethod
    def init_results_pool(results_data):
        GEDICorrect._shared_results = {df.iloc[0]['shot_number']: df
                                       for df in results_data if not df.empty}

    @staticmethod
    def clear_results_pool():
        global _shared_results
        GEDICorrect._shared_results = None

    def simulate(self, grid_size=15, grid_step=1, n_points=100, max_radius=12.5, min_dist=1.0):
        """
        This function performs the correction sequence on given GEDI files to class instance

        Args:
            n_points: Number of points to simulate around original footprint
            max_radius: Maximum radius distance to place points
            min_dist: Minimum distance between each simulated point

        Returns:
            None
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

        ## Save ALS list of files for Simulation
        with open(os.path.join(self.temp_dir.name, "alsList.txt"), mode="w") as las_txt_file:
            for als_file in als_list:
                las_txt_file.write(f"{als_file}\n")

        ## Always simple a middle point
        if not grid_size % 2:
            grid_size += 1
        
        ## Mode Selection
        if self.mode == "orbit":
            print("[Setup] Correcting at the ORBIT LEVEL")
            self._orbit_simulate(grid_size=grid_size, grid_step=grid_step)

        if self.mode == "beam":
            print("[Setup] Correcting at the BEAM LEVEL")
            self._beam_simulate(grid_size=grid_size, grid_step=grid_step)

        if self.mode == "footprint":
            print("[Setup] Correcting at the FOOTPRINT LEVEL")
            self._footprint_simulate(grid_size=grid_size, grid_step=grid_step)

        # Clean Temp Dir
        del self.temp_dir


    def _setup(self):
        """
        Sanity check and setup of GEDI Files and ALS Files

        Args:
            None

        Returns: 
            bool: Setup Status, if true, the correction can continue
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

        # Setup complete and OKAY
        return True
            
    def _check_setup_status(self):
        """
        Check Setup Status

        returns:
            bool: Setup Status
        """
        return self.setup_status

    def _save_outputs(self, results, filename, cluster_results=None, offset=None, beam_offset=None) -> None:
        """
        Saves the output of the correction process into new files of GEDI footprints.

        Consists of three output modes:
            1 - Output a file of simulations at original location of each GEDI footprint;
            2 - Output a file of all simulations around each GEDI footprint;
            3 - Output a file of the corrected (highest scored) simulated footprints.

        Args:
            results (list): A list of DataFrames, each containing scored simulated GEDI footprints for a
                            specific footprint (output of the scoring unit).
            filename (str): Filename of output GEDI granule.
            offset (tuple): A tuple containing the offset coordinates to move each footprint in results. Valid
                            for Orbit-level correction
            beam_offset (dict): A dictionary containing the BEAM and offset tuple pairing, which represents a
                                specific offset for each BEAM. Valid for Beam-level correction.

        Returns:
            None
        """

        if self.save_origin_location:
            # Save simulated footprint at original location
            origin_loc = []
            for gpd_df in results:
                if len(gpd_df) != 0:
                    # RXWAVECOUNT array to str for output purposes
                    gpd_df['RXWAVECOUNT'] = gpd_df['RXWAVECOUNT'].astype(str)

                    origin_loc.append(gpd_df.loc[0])

            # Save original location dataframe
            origin_df = gpd.GeoDataFrame(origin_loc, crs=self.crs).set_geometry('geometry')
            out_filename = filename.split('/')[-1]
            origin_df = origin_df.drop(columns=['FSIGMA'])
            origin_df.to_file(os.path.join(self.out_dir, 'ORIGINLOC_'+out_filename))


        if self.save_sim_points:
            # Save sim points to SHP file
            save_df = []
            for gpd_df in results:
                if len(gpd_df) != 0:
                    # RXWAVECOUNT array to str for output purposes
                    gpd_df['RXWAVECOUNT'] = gpd_df['RXWAVECOUNT'].apply(str)
                    
                    if not self.random:
                        gpd_df['grid_offset'] = gpd_df['grid_offset'].apply(str)
                        gpd_df['cluster_bounds'] = gpd_df['cluster_bounds'].apply(str)

                    save_df.append(gpd_df)

            # Append to to be saved dataframe
            sim_save_df = gpd.GeoDataFrame(pd.concat(save_df))
            sim_save_df.crs = self.crs
            sim_save_df = sim_save_df.drop(columns=['FSIGMA']) 
            sim_save_out_filename = filename.split('/')[-1]
            sim_save_df.to_file(os.path.join(self.out_dir, 'SIMPOINTS_'+sim_save_out_filename))

        ## Save correct (highest scored) simulated footprints
        if offset:
            # Orbit-Level mode
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

            out_df = gpd.GeoDataFrame(selected_rows, crs=self.crs, geometry='geometry')
            out_df = out_df.drop(columns=['FSIGMA'])
            save_out_filename =  filename.split('/')[-1]
            out_df.to_file(os.path.join(self.out_dir, 'ORBIT_'+save_out_filename))

        elif beam_offset:
            # Beam-Level mode
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
            save_out_filename = filename.split('/')[-1]
            out_df.to_file(os.path.join(self.out_dir, 'BEAM_' + save_out_filename))

        else:
            # Footprint-Level mode
            # Already selected best footprints in a geodataframe
            if not cluster_results is None:
                cluster_results['RXWAVECOUNT'] = cluster_results['RXWAVECOUNT'].apply(str)
                cluster_results['grid_offset'] = cluster_results['grid_offset'].apply(str)
                cluster_results['cluster_bounds'] = cluster_results['cluster_bounds'].apply(str)
                cluster_results.drop(columns=['FSIGMA'], inplace=True)

                ## Save corrected footprints to GPKG
                #out_df = gpd.GeoDataFrame(final_df, crs=self.crs).set_geometry('geometry')
                out_filename = filename.split('/')[-1]
                cluster_results.to_file(os.path.join(self.out_dir, 'CORRECTED_'+out_filename))
                del cluster_results
                gc.collect()


    def score_clusters_sequential(cluster_dict, results_dict):
        """
        Sequential version of cluster scoring. Calculates the mean scores of each grid offset for each cluster
        which then finds the best offset for each footprint.
        
        Args:
            cluster_dict (dict): Dictionary where keys are main shot_numbers and values are lists of cluster member shot_numbers.
            results_dict (dict): Dictionary mapping shot_numbers to their corresponding GeoDataFrames.

        Returns:
            list: List of best-scored GeoDataFrame rows (one per cluster).
        """
        output_rows = []

        # For every cluster
        for main_sn, cluster_sns in cluster_dict.items():

            # there will be a score accumulator
            score_accumulator = {}

            # which is then updated for each offset of the grid with its final_score
            for sn in cluster_sns:
                df = results_dict.get(sn)
                if df is None or len(df) <= 1:
                    continue
                for _, row in df.iterrows():
                    offset = row['grid_offset']
                    score = row['final_score']
                    score_accumulator.setdefault(offset, []).append(score)

            if not score_accumulator:
                continue

            # Compute mean scores rounded to 5 decimals of the cluster
            mean_scores = {
                offset: round(sum(scores) / len(scores), 5)
                for offset, scores in score_accumulator.items()
            }

            # Get best offset of the cluster
            best_offset = max(mean_scores.items(), key=lambda x: x[1])[0]

            # Set best offset of the cluster for the main footprint of the cluster
            main_df = results_dict.get(main_sn)
            if main_df is None or len(main_df) <= 1:
                continue
            
            # Move centroid footprint to its cluster best offset
            corrected_row = main_df[main_df['grid_offset'] == best_offset]

            if not corrected_row.empty:
                output_rows.append(corrected_row.iloc[0])

        return output_rows


    @staticmethod
    def score_cluster_parallel(cluster):
        """
        Parallel version of cluster scoring. Calculates the mean scores of each grid offset for each cluster
        which then finds the best offset for each footprint.

        Args:
            cluster (tuple): Cluster tuple consisting of (Main footprint shot_number, List of neighboring Shot_numbers)

        Returns:
            pd.Series or None: The row with the highest mean 'final_score' for corresponding Main Shot_number, or None if footprint does not exist.
        """
        main_sn, cluster_sns = cluster
        score_accumulator = {}

        for sn in cluster_sns:
            df = GEDICorrect._shared_results.get(sn)
            if df is None or len(df) <= 1:
                continue
            for _, row in df.iterrows():
                offset = row['grid_offset']
                score = row['final_score']
                score_accumulator.setdefault(offset, []).append(score)

        if not score_accumulator:
            return None

        mean_scores = {
            offset: round(sum(scores) / len(scores), 5)
            for offset, scores in score_accumulator.items()
        }

        best_offset = max(mean_scores.items(), key=lambda x: x[1])[0]
        main_df = GEDICorrect._shared_results.get(main_sn)

        if main_df is None or len(main_df) <= 1:
            return None

        corrected_row = main_df[main_df['grid_offset'] == best_offset]

        if not corrected_row.empty:
            return corrected_row.iloc[0]

        return None
    


    def _footprint_simulate(self, grid_size=15, grid_step=1, time_window=0.215):
        """
        Simulates and Scores at the footprint-level all of the input GEDI granules.
        User can select parallelization. After processing every single input GEDI orbit
        it outputs all simulated, original and corrected (highest scored) footprints in
        files using the '_save_outputs()' function.

        For this footprint-level approach, it uses a "Clustering" Algorithm, which clusters
        every footprint with neighboring footprints according to the ISS vibration rate.

        Args:
            grid_size (int): Size of search grid around each reported footprint. Final size of grid is
                             'Grid_Size x Grid_Size'
            grid_step (int): Distance (in meters) between each point in grid. Defaults to 1 meter.

            time_window (float): Vibration rate in Hz of the GEDI platform. Defaults to 0.215 Hz.

        Returns:
            None
        """
        
        # Generate search grid
        offsets = generate_grid(x_max=grid_size, y_max=grid_size, step=grid_step)

        # Loop (Simulation -> Scoring -> Output) for each input GEDI file
        for filename, footprint_df in self.gedi_granules.items():
            print(f"[Simulate] Correcting granule {filename}")

            # Spawn Scorer
            scorer = CorrectionScorer(original_df=footprint_df, crs=self.crs, criteria=self.criteria)
            footprints = [row for _, row in footprint_df.iterrows()]

            processed_footprints = []
            results = []

            if self.use_parallel:
                print(f"[Simulate] Running in parallel mode with {self.n_processes} processes")
                with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:
                    
                    partial_func_processing = partial(process_all_footprints,
                                                    temp_dir=self.temp_dir.name,
                                                    las_dir=self.las_dir,
                                                    original_df=footprint_df,
                                                    crs=str(self.crs).split(":")[-1],
                                                    grid=offsets)

                    with tqdm(total=len(footprints), desc="Processing Footprints") as pbar:
                        for processed in pool.imap_unordered(partial_func_processing, footprints):
                            processed_footprints.append(processed)
                            pbar.update(1)

                filtered_processed_footprints = [f for f in processed_footprints if not (len(f) == 1 and type(f[0]) == int)]
                del processed_footprints

                partial_func_correction = partial(scorer.score)
                with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:
                    with tqdm(total=len(filtered_processed_footprints), desc="Correcting Footprints") as pbar:
                        for corrected in pool.imap_unordered(partial_func_correction, filtered_processed_footprints):
                            results.append(corrected)
                            pbar.update(1)

                results = [df for df in results if len(df) > 1]

                del partial_func_correction
                gc.collect()

            else:
                print(f"[Simulate] Running in sequential mode")

                footprints = [row for row in footprints]

                for fpt in tqdm(footprints, desc="Processing Footprints"):
                    processed_footprints.append(process_all_footprints(
                        fpt, self.temp_dir.name,
                        las_dir=self.las_dir,
                        original_df=footprint_df,
                        crs=str(self.crs).split(":")[-1],
                        grid=offsets))

                filtered_processed_footprints = [f for f in processed_footprints if not (len(f) == 1 and type(f[0]) == int)]

                for fpt in tqdm(filtered_processed_footprints, desc="Correcting Footprints"):
                    if len(fpt) > 1:
                        results.append(scorer.score(fpt))

            # Sort final simulated and scored list by shot_number
            results = sorted(results, key=lambda df: df.iloc[0]['shot_number'])

            # Build small dataframe for the clusterization process
            cluster_df = pd.DataFrame([
            {
                'shot_number': df.iloc[0]['shot_number'],
                'geolocation_delta_time': df.iloc[0]['geolocation_delta_time'],
                'BEAM': df.iloc[0]['BEAM']
            }
                for df in results if len(df) > 1
            ])

            cluster_df = cluster_df.sort_values(by=["geolocation_delta_time", "BEAM", "shot_number"]).reset_index(drop=True)

            # Find clusters for each footprint in the already filtered, simulated and scored footprint dataset
            clusters_dict = cluster_footprints(cluster_df, time_window=time_window)

            del cluster_df

            # Add information about clusters to final dataframe
            results = annotate_clusters(results, clusters_dict)
            cluster_args = [(main_sn, members) for main_sn, members in clusters_dict.items()]
            del clusters_dict

            # Find optimal score for each cluster
            if self.use_parallel:
                with multiprocessing.Pool(processes=self.n_processes, maxtasksperchild=5,
                                          initializer=GEDICorrect.init_results_pool,
                                          initargs=(results,)) as pool:
                    cluster_results = list(tqdm(
                        pool.imap_unordered(GEDICorrect.score_cluster_parallel, cluster_args),
                        total=len(cluster_args),
                        desc="Clustering Footprints"
                    ))

                GEDICorrect.clear_results_pool()
            else:
                # Use sequential
                results_dict = {df.iloc[0]['shot_number']: df 
                                for df in results if not df.empty}

                cluster_results = self.score_clusters_sequential(cluster_args, results_dict)

            gc.collect()

            # Add information of final offset for each footprint
            corrected_clusters = gpd.GeoDataFrame([fpt for fpt in cluster_results if fpt is not None], crs=self.crs, geometry='geometry')
            corrected_clusters = add_cluster_stats(corrected_clusters)

            print(f"[Simulate] Saving corrected granule {filename}")
            self._save_outputs(results, filename, cluster_results=corrected_clusters)

            del footprint_df, scorer, footprints, filtered_processed_footprints
            del cluster_args, corrected_clusters, results, cluster_results

            for p in multiprocessing.active_children():
                p.terminate()
                p.join()

            gc.collect()

    def _process_orbit_level(self, footprint, grid, temp_dir, original_df, filename, crs, scorer, score_dict, lock):
        '''
        Helper function for the partial used for both parallel and sequential modes for the
        orbit-level correction at '_orbit_simulate()'. Grabs each footprint and performs simulation
        and scoring, returning a DataFrame containing all of the simulated points. It then
        updates a global variable (controlled by the lock if using parallelization) based on
        the entire orbit's best offset.

        Args:
            footprint (DataFrame): Single footprint entry of a Dataframe, containing 
                                   information and relevant variables
            grid (list): A list of all possible offsets of the given grid
            temp_dir (TemporaryDirectory): A temporary directory to keep information
                                           and I/O operations during simulation
            original_df (DataFrame): The original (reported GEDI) dataframe used in the
                                     Scoring process
            filename (str): Original GEDI granule filename
            crs (pyproj.CRS): Coordinate Reference System of both ALS and GEDI, used for simulation
            scorer (CorrectionScorer): CorrectionScorer instance used to score simulated points
            score_dict (dict): A dictionary containing pairs of offset (tuple) and best calculated score for
                               that offset. Used as a global variable to be updated by all processes (or 1 process)
            lock (Manager.Lock): Lock instance to control access to the global variable 'score_dict'. Created by
                                 Manager().

        Returns:
            scored_df (DataFrame): A dataframe of all of the simulated points around given 'footprint'. 
                                   Each simulation also has its respective score.
        '''
         
        # Simulate
        simulated_df = process_all_footprints(footprint, temp_dir, self.las_dir, original_df, crs, grid=grid)

        # Then Score
        scored_df = scorer.score(simulated_df)

        if len(scored_df) == 0:
            return []

        # Return scored simulation
        temp_score_dict = scored_df.set_index('grid_offset')['final_score'].to_dict()

        # Update the global score_dict in a thread-safe way
        if not lock is None:
            with lock:
                for offset, score in temp_score_dict.items():
                    score_dict[offset] = score_dict.get(offset, 0) + score
        else:
            # For sequential mode
            for offset, score in temp_score_dict.items():
                score_dict[offset] = score_dict.get(offset, 0) + score

        return scored_df


    def _process_beam_level(self, footprint, grid, temp_dir, original_df, crs, scorer, score_dict, lock):
        '''
        Helper function for the partial used for both parallel and sequential modes for the
        beam-level correction at '_beam_simulate()'. Grabs each footprint and performs simulation
        and scoring, returning a DataFrame containing all of the simulated points. It then
        updates a global variable (controlled by the lock if using parallelization) based on
        each BEAM's best offset.

        Args:
            footprint (DataFrame): Single footprint entry of a Dataframe, containing 
                                   information and relevant variables
            grid (list): A list of all possible offsets of the given grid
            temp_dir (TemporaryDirectory): A temporary directory to keep information
                                           and I/O operations during simulation
            original_df (DataFrame): The original (reported GEDI) dataframe used in the
                                     Scoring process
            filename (str): Original GEDI granule filename
            crs (pyproj.CRS): Coordinate Reference System of both ALS and GEDI, used for simulation
            scorer (CorrectionScorer): CorrectionScorer instance used to score simulated points
            score_dict (dict): A dictionary containing pairs of offset (tuple) and best calculated score for
                               that offset. Used as a global variable to be updated by all processes (or 1 process)
            lock (Manager.Lock): Lock instance to control access to the global variable 'score_dict'. Created by
                                 Manager().

        Returns:
            scored_df (DataFrame): A dataframe of all of the simulated points around given 'footprint'. 
                                   Each simulation also has its respective score.
        '''

        # Simulate
        simulated_df = process_all_footprints(footprint, temp_dir, self.las_dir, original_df, crs, grid=grid)

        # Then Score
        scored_df = scorer.score(simulated_df)

        # Catch errors / invalid scored footprints
        if len(scored_df) == 0:
            return []
        
        # Get BEAM name
        beam_id = scored_df['BEAM'].values[0]

        # Return scored simulation
        temp_score_dict = scored_df.set_index('grid_offset')['final_score'].to_dict()

        # Update global offset scores
        if not lock is None:
            with lock:
                for offset, score in temp_score_dict.items():
                    score_dict[beam_id][offset] = score_dict[beam_id].get(offset, 0) + score  # Accumulate the score
        else:
            # For sequential mode
            for offset, score in temp_score_dict.items():
                score_dict[beam_id][offset] = score_dict[beam_id].get(offset, 0) + score  # Introduce newly seen score

        return scored_df
 

    def _orbit_simulate(self, grid_size, grid_step):
        '''
        Simulates and Scores at the orbit-level all of the input GEDI granules.
        User can select parallelization. After processing every single input GEDI orbit
        it outputs all simulated, original and corrected (highest scored) footprints in
        files using the '_save_outputs()' function.

        Args:
            grid_size (int): Size of search grid around each reported footprint. Final size of grid is
                             'Grid_Size x Grid_Size'
            grid_step (int): Distance (in meters) between each point in grid. Defaults to 1 meter.

        Returns:
            None
        '''
        score_dict = {} if not self.use_parallel else Manager().dict()
        lock = None if not self.use_parallel else Manager().Lock() # Used to lock global variable of best offset

        # Iterate through each gedi file
        for filename, footprint_df in self.gedi_granules.items():
            print(f"[Simulate] Correcting granule {filename}")
            print(f"[Simulate] Criteria: {self.criteria}")

            scorer = CorrectionScorer(original_df=footprint_df, crs=self.crs, criteria=self.criteria) # Init Scorer
            offsets = generate_grid(x_max=grid_size, y_max=grid_size, step=grid_step) # Generate grid

            footprints = [row for i, row in footprint_df.iterrows()]
            processed_footprints = []

            if self.use_parallel:
                # Parallelization
                print(f"[Simulate] Running in parallel mode with {self.n_processes} processes...")
                with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:

                    partial_func_processing = partial(self._process_orbit_level,
                                                    grid=offsets,
                                                    temp_dir=self.temp_dir.name,
                                                    original_df=footprint_df,
                                                    filename=filename,
                                                    crs=str(self.crs).split(":")[-1],
                                                    scorer=scorer,
                                                    score_dict=score_dict,
                                                    lock=lock)

                    with tqdm(total=len(footprints), desc="Processing Points") as pbar:
                        for correct_fpt in pool.imap_unordered(partial_func_processing, footprints):
                            processed_footprints.append(correct_fpt)
                            pbar.update(1)
            else:
                print("[Simulate] Running in sequential mode...")
                with tqdm(total=len(footprints), desc="Processing Footprints") as pbar:
                    for footprint in footprints:
                        processed = self._process_orbit_level(
                            footprint, offsets, self.temp_dir.name, footprint_df,
                            filename, str(self.crs).split(":")[-1], scorer, score_dict, lock
                        )
                        processed_footprints.append(processed)
                        pbar.update(1)

            processed_footprints = [x for x in processed_footprints if not type(x) is list]

            # After all footprints are processed, calculate the mean score
            num_footprints = len(processed_footprints)

            if num_footprints == 0:
                print(f"{filename} contains invalid footprints for processing. Skipping...")
                continue

            mean_score_dict = {offset: score / num_footprints for offset, score in score_dict.items()}

            # Select the best offset based on the highest mean score
            best_offset = max(mean_score_dict, key=mean_score_dict.get)

            print(f"Best offset: {best_offset} with mean score: {mean_score_dict[best_offset]}")

            # Save files after correcting
            print(f"[Simulate] Saving corrected granule {filename}")
            self._save_outputs(processed_footprints, filename, offset=best_offset)

            del score_dict

    def _beam_simulate(self, grid_size, grid_step):
        '''
        Simulates and Scores at the Beam-level all of the input GEDI granules.
        User can select parallelization. After processing every single input GEDI orbit
        it outputs all simulated, original and corrected (highest scored) footprints in
        files using the '_save_outputs()' function.

        Args:
            grid_size (int): Number of points to simulate around each reported footprint
            grid_step (float): Maximum radius (distance in meters) from reported footprint to simulate points.

        Returns:
            None
        '''

        # Initialize multiprocessing resources if running in parallel
        manager = Manager() if self.use_parallel else None
        lock = manager.Lock() if self.use_parallel else None

        for filename, footprint_df in self.gedi_granules.items():
            print(f"[Simulate] Correcting granule {filename}")
            print(f"[Simulate] Criteria: {self.criteria}")

            scorer = CorrectionScorer(original_df=footprint_df, crs=self.crs, criteria=self.criteria)
            offsets = generate_grid(x_max=grid_size, y_max=grid_size, step=grid_step)

            footprints = [row for i, row in footprint_df.iterrows()]
            processed_footprints = []

            # Define all valid GEDI BEAM IDs
            beam_ids = ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011", 
                        "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
            
             # Initialize score dictionary
            score_dict = manager.dict() if self.use_parallel else {}
            for beam_id in beam_ids:
                score_dict[beam_id] = manager.dict() if self.use_parallel else {}
        
            # Prepare function for execution
            partial_func_processing = partial(self._process_beam_level,
                                                  grid=offsets,
                                                  temp_dir=self.temp_dir.name,
                                                  original_df=footprint_df,
                                                  crs=str(self.crs).split(":")[-1],
                                                  scorer=scorer,
                                                  score_dict=score_dict,
                                                  lock=lock)

            # Run in parallel or sequentially
            if self.use_parallel:
                with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:
                    with tqdm(total=len(footprints), desc="Processing Points") as pbar:
                        for correct_fpt in pool.imap_unordered(partial_func_processing, footprints):
                            processed_footprints.append(correct_fpt)
                            pbar.update(1)
            else:
                for footprint in tqdm(footprints, desc="Processing Points"):
                    processed_footprints.append(partial_func_processing(footprint))

            processed_footprints = [x for x in processed_footprints if not type(x) is list]

            best_beam_offset = {}
            beam_counts = {}

            # Get each BEAM count
            for fpt in processed_footprints:
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
            self._save_outputs(processed_footprints, filename, beam_offset=best_beam_offset)

            del score_dict