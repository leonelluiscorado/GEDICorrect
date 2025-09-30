"""
Implementation of GEDICorrect class
"""

import os
import tempfile
import geopandas as gpd
import pandas as pd
from collections import defaultdict

from .data_process import *
from .simulation import process_all_footprints, init_random_seed, process_all_footprints
from .scorer import CorrectionScorer
from .dataclass import ScoredFootprint

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
                 als_crs=None,
                 als_algorithm='convex',
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
        self.als_crs = str(als_crs).split(":")[-1] if als_crs else None
        self.als_algorithm = als_algorithm

        self.gedi_granules = {}

        # Perform setup check: ALS bounding and GEDI
        self.setup_status = self._setup()


    def simulate(self, grid_size=15, grid_step=1):
        """
        This function performs the correction sequence on given GEDI files to class instance

        Args:
            grid_size: Size of the grid to determine the number of points to simulate around original footprint
            grid_step: Step size of grid points around original footprint

        Returns:
            None
        """

        if not self.setup_status:
            # Class not properly setup
            raise Exception("[Simulate] Class not properly setup due to error in GEDI Granules or",
                              " opening LAS files. You must create another instance of this class")

        ## Create temporary directory for temp files
        self.temp_dir = tempfile.TemporaryDirectory()
        print(f"[Simulate] Saving files in temporary directory : {self.temp_dir.name}")

        ## Save ALS List from *las_dir* to temporary directory for simulation
        als_list = [os.path.join(self.las_dir, f) for f in os.listdir(self.las_dir) if (f.endswith('.las') or f.endswith('.laz'))]

        ## Save ALS list of files for Simulation
        with open(os.path.join(self.temp_dir.name, "alsList.txt"), mode="w") as las_txt_file:
            for als_file in als_list:
                las_txt_file.write(f"{als_file}\n")

        ## Always sample a middle point on the grid
        if not grid_size % 2:
            grid_size += 1
        
        # Generate grid of points
        offsets = generate_grid(grid_size, grid_size, step=grid_step)

        # Correct every input file
        for filename, footprint_df in self.gedi_granules.items():

            # Spawn Scorer for dataset
            scorer = CorrectionScorer(original_df=footprint_df, crs=self.crs, criteria=self.criteria)

            ## Simulate and Score all points
            processed_footprints = self._sim_and_score(footprint_df=footprint_df, scorer=scorer, offsets=offsets)

            # Select correction mode
            if self.mode == "orbit":
                print("[GEDICorrect] Correcting at the ORBIT LEVEL")
                best_offsets = self._orbit_correct(processed_footprints)

            if self.mode == "beam":
                print("[GEDICorrect] Correcting at the BEAM LEVEL")
                best_offsets = self._beam_correct(processed_footprints)

            if self.mode == "footprint":
                print("[GEDICorrect] Correcting at the FOOTPRINT LEVEL")
                best_offsets = self._footprint_correct(processed_footprints)

            if not len(best_offsets):
                print(f"[GEDICorrect] Error in correcting file {filename}. Try again.")
                continue

            ## Output step
            # Resimulate best offset points for output
            corrected_rows = self._resimulate_best_offsets(best_offsets, footprint_df, scorer)

            # Save corrected file
            self._save_outputs(results=corrected_rows, filename=filename)

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
            self.las_extents, self.crs = get_las_extents(las_files_dir=self.las_dir, explicit_epsg=self.als_crs, algorithm=self.als_algorithm)
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
            2 - Output a file of the corrected (highest scored) simulated footprints.

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

        rows = []

        # Transform each footprint dataframe in a format that GeoDataFrame expects
        for df in results:
            if df.empty:
                continue
            row = df.iloc[0]
            rows.append(row.to_dict()) ## to dictionary

        if rows:
            out_df = gpd.GeoDataFrame(rows, crs=self.crs, geometry="geometry")

            out_df["RXWAVECOUNT"] = out_df["RXWAVECOUNT"].astype(str)
            out_df["grid_offset"] = out_df["grid_offset"].astype(str)

            out_df.drop(columns=["FSIGMA"], inplace=True, errors="ignore") # Drop FSIGMA Column (incompatible with final Dataframe)

            filename = filename.split("/")[-1]

            if self.mode == "orbit":
                out_df.to_file(os.path.join(self.out_dir, f"ORBIT_{filename}"))

            if self.mode == "beam":
                out_df.to_file(os.path.join(self.out_dir, f"BEAM_{filename}"))

            if self.mode == "footprint":
                out_df.to_file(os.path.join(self.out_dir, f"FOOTPRINT_{filename}"))

        return


    def _orbit_correct(self, summaries):
        '''
        Selects the best offset at the orbit-level for a (Simulated and Scored) GEDI file.

        Args:
            grid_size (int): Size of search grid around each reported footprint. Final size of grid is
                             'Grid_Size x Grid_Size'
            grid_step (int): Distance (in meters) between each point in grid. Defaults to 1 meter.

        Returns:
            None

        '''
        per_offset_scores = defaultdict(list)
        for summary in summaries:
            for offset, score in summary.offset_scores:
                per_offset_scores[offset].append(score)

        if not per_offset_scores:
            return {}

        best_offset = max(
            per_offset_scores.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
        )[0]

        return {summary.shot_number: best_offset for summary in summaries}
    
    def _beam_correct(self, footprints):
        """Select the best offset at the beam-level for each footprint inside a beam"""
        
        # Build BEAM dictionary
        per_beam = defaultdict(lambda: defaultdict(list))

        for footprint in footprints:
            for offset, score in footprint.offset_scores:
                per_beam[footprint.beam][offset].append(score)

        if not per_beam:
            return {}

        beam_best_offset = {}
        for beam, scores_dict in per_beam.items():
            if not scores_dict:
                continue
            beam_best_offset[beam] = max(
                scores_dict.items(),
                key=lambda item: sum(item[1]) / len(item[1]),
            )[0]

        footprint_offsets = {}
        for footprint in footprints:
            offset = beam_best_offset.get(footprint.beam)
            if offset is not None:
                footprint_offsets[footprint.shot_number] = offset

        return footprint_offsets
    

    def _footprint_correct(self, footprints):
        """Select the best offset at the footprint-level using time-window clustering"""

        if not footprints:
            return {}

        # Build a dataframe the clustering function expects
        cluster_df = pd.DataFrame(
            {
                "shot_number": [str(s.shot_number) for s in footprints],
                "geolocation_delta_time": [float(s.delta_time) for s in footprints],
                "BEAM": [str(s.beam) for s in footprints],
            }
        ).sort_values(
            ["geolocation_delta_time", "BEAM", "shot_number"]
        ).reset_index(drop=True)

        # Cluster footprints by time_window
        clusters = cluster_footprints(cluster_df, time_window=self.time_window)
        clustered_rows = annotate_clusters(footprints, clusters)
        score_cache = {int(s.shot_number): s.offset_scores for s in footprints}

        best_offsets = {}
        # Average per offset inside each cluster
        for main_sn, members in clusters.items():
            accumulator = defaultdict(list)
            for member in members:
                for offset, score in score_cache.get(int(member), []):
                    accumulator[offset].append(score)

            if not accumulator:
                continue

            best_offset = max(
                accumulator.items(),
                key=lambda item: sum(item[1]) / len(item[1])
            )[0]

            best_offsets[int(main_sn)] = best_offset

        # Footprints not assigned to a cluster (singletons) can fall back to their own best score
        for footprint in footprints:
            shot_id = int(footprint.shot_number)
            if shot_id not in best_offsets and footprint.offset_scores:
                best_offsets[shot_id] = max(footprint.offset_scores, key=lambda entry: entry[1])[0]

        return best_offsets
    

    def _sim_and_score(self, footprint_df, scorer, offsets):
        '''
        Simulates and scores all footprints inside a file.
        Saves corrected footprints in the simplest form, with only a shot_number, beam,
        geolocation_delta_time and every final_score for each offset, calculated in the Scoring Step
        '''
        scored_footprints = []

        partial_func_processing = partial(
            self.__sim_scorer_footprint,
            scorer=scorer,
            offsets=offsets,
            original_df=footprint_df,
        )

        fpt_rows = [row for _, row in footprint_df.iterrows()]

        if self.use_parallel:
            with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:
                for corrected_footprint in tqdm(pool.imap_unordered(partial_func_processing, fpt_rows),
                                                total=len(fpt_rows),
                                                desc=f"Processing footprints"):
                    if corrected_footprint:
                        scored_footprints.append(corrected_footprint)

        else:
            for fpt in tqdm(fpt_rows, desc=f"Processing footprints"):
                corrected_footprint = partial_func_processing(fpt)
                if corrected_footprint:
                        scored_footprints.append(corrected_footprint)

        return scored_footprints
    
    
    def __sim_scorer_footprint(self, footprint_row, scorer, offsets, original_df):
        '''
        Helper/Partial Function used in processing of '_sim_and_score'
        '''

        sim = process_all_footprints(
            footprint_row,
            temp_dir=self.temp_dir.name,
            las_dir=self.las_dir,
            original_df=original_df,
            crs=str(self.crs).split(":")[-1],
            grid=offsets,
        )

        if not isinstance(sim, pd.DataFrame) or len(sim) < 1:
            return None

        scored = scorer.score(sim)
        if len(scored) < 1:
            return None

        shot = int(scored["shot_number"].iloc[0])
        beam = str(scored["BEAM"].iloc[0])
        delta = float(scored["geolocation_delta_time"].iloc[0])
        offset_scores = list(zip(scored["grid_offset"].tolist(), scored["final_score"].tolist()))

        return ScoredFootprint(shot_number=shot, beam=beam, delta_time=delta, offset_scores=offset_scores)
    

    def _resimulate_best_offsets(self, best_offsets, original_df, scorer):
        '''
        Resimulator used before '_save_outputs'. Simulates GEDI footprints at their best scored location offsets.
        '''

        footprints_to_resimulate = [
            (shot_number, offset)
            for shot_number, offset in best_offsets.items()
        ]

        ### Get intersecting las for each shot_number

        single_resimulator = partial(
            self.__resimulate_single,
            original_df=original_df,
            scorer=scorer,
            temp_dir=self.temp_dir.name,
        )

        corrected_rows = []

        if self.use_parallel:
            with multiprocessing.Pool(self.n_processes, maxtasksperchild=5, initializer=init_random_seed) as pool:
                for result in tqdm(pool.imap_unordered(single_resimulator, footprints_to_resimulate),
                                   total=len(footprints_to_resimulate),
                                   desc="Re-simulating for output"):
                    if result is not None:
                        corrected_rows.append(result)
        else:
            for footprint in tqdm(footprints_to_resimulate, desc="Re-simulating for output"):
                result = single_resimulator(footprint)
                if result is not None:
                    corrected_rows.append(result)

        return corrected_rows
    

    def __resimulate_single(self, footprint, original_df, scorer, temp_dir):
        '''
        Helper/Partial Function for '_resimulate_best_offsets'
        Simulates and Scores a given footprint of unique shot_number and offset
        '''

        shot_number, offset = footprint
        base_row = original_df.loc[original_df['shot_number_x'] == shot_number]

        if base_row.empty:
            print(f"[Resimulate] Target footprint {shot_number} does not exist in the original file.")
            return None

        footprint = base_row.iloc[0]

        single_simulation = process_all_footprints(
            footprint,
            temp_dir=temp_dir,
            las_dir=self.las_dir,
            original_df=original_df,
            crs=str(self.crs).split(":")[-1],
            grid=[offset],
            simulate_original=False,
        )

        if not isinstance(single_simulation, pd.DataFrame) or len(single_simulation) < 1:
            return None

        scored_single = scorer.score(single_simulation)
        if len(scored_single) < 1:
            return None
        
        # Footprint will have a 'final_score' of 1.0, since it's the only one footprint to average.
        # Drop the final score column
        # TODO: Display real final_score for that footprint.
        scored_single = scored_single.drop(columns=["final_score"], errors="ignore")

        return scored_single