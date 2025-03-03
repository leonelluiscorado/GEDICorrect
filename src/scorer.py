"""
Implementation of the Scorer class
"""

import pandas as pd
import geopandas as gpd
import numpy as np

import math
from scipy.stats import pearsonr, spearmanr, norm, ecdf
from scipy.signal import correlate
from .waveform_processing import *
from .data_process import clean_cols_rh
from .metric import *
import matplotlib.pyplot as plt
import matplotlib

possible_criteria = ['wave_pearson', 'wave_spearman', 'wave_distance', 'kl', 'rh_distance', 'terrain']

class CorrectionScorer:
    """
    The CorrectionScorer :class: calculates different select metrics for GEDI footprint geolocation correction
    Args:
        original_df : GeoDataFrame containing original, to be processed, footprint information to compare to the simulated
        criteria    : A string containing the desired metrics for calculation. 'all' selects possible_criteria
        ground      : Option between 'GEDI' and 'ALS' for the "Terrain_Matching" Criteria, whether choose the ALS ground calculated by the program or
                      the original GEDI ground elevation
        product     : Option between 'L1B' and 'L2A' of the GEDI product for the "Terrain_Matching" Criteria, whether choose the geolocation_lastbin
                      or lowestmode
        add_info    : Flag as to whether add info (difference, mean) about every metric defined in 'criteria'
    """

    def __init__(self, original_df=None, criteria='all', ground='GEDI', product='L2A', add_info=True):
        self.original_df = original_df
        self.criteria = []
        self.ground = ground
        self.product = product
        self.add_info = add_info

        self.set_criteria(criteria)

        # Debug
        self.debug = True

    def set_criteria(self, criteria):
        """
        Setter function for criteria.
        """
        try:
            self.criteria = self._parse_criteria(criteria)
        except:
            print(f"[Correction] Criterion {criteria} not valid. Please select between [rh , rh_correlation , wave , terrain]. Only one correlation method can be choosed.")
            self.criteria = []

    def _parse_criteria(self, criteria):
        """
        Parses criteria variable string into a list of options.

        Example argument would be: criteria = "rh rh_correlation wave"
        """
        if criteria == 'all':
            return possible_criteria

        criteria = criteria.split(" ")

        for criterion in criteria:
            if not criterion in possible_criteria:
                raise ValueError
        
        # Both correlations cannot be chosen
        if "wave_pearson" in criteria and "wave_spearman" in criteria:
            raise ValueError

        return criteria


    def score(self, sim_footprints):
        """
        Scores a list of simulated footprints that align with the original footprint shot_number from the original_df and calculates
        from selected criteria at class creation
        """

        # Do not score invalid simulated footprints
        if len(sim_footprints) <= 1:
            return []

        # No criteria selected
        if len(self.criteria) == 0:
            print("[Correction] Criteria list set to 0. Please list selected criteria with a 'set_criteria' function call on this object.")
            return []

        sim_footprints = sim_footprints.copy()

        # Match original footprint shot_number
        footprint_shot_number = sim_footprints['shot_number'][0]
        original_footprint = self.original_df.loc[self.original_df['shot_number_x'] == int(footprint_shot_number)]
        sim_footprints['BEAM'] = original_footprint['BEAM'].values[0]
        
        # Rh Matching
        sim_footprints = self._rh_distance(original_footprint, sim_footprints)

        # Terrain Matching
        sim_footprints = self._terrain_matching(original_footprint, sim_footprints)

        # Waveform Matching
        sim_footprints = self._waveform_matching(original_footprint, sim_footprints)

        if len(sim_footprints) == 0:
            return []

        # Add L2A rh metrics to final file
        if self.add_info:
            rh_col_types = [f'{i}' for i in range(25, 105, 5)]

            for rh in rh_col_types:
                sim_footprints[f'rh{rh}'] = original_footprint[f'rh_{rh}'].values[0]

        # Final score for selection
        sim_footprints['final_score'] = self._calculate_score(sim_footprints)

        return sim_footprints


    def _rh_correlation(self, original, simulated):
        """
        ///// PROPOSAL METRIC - NOT TESTED /////
        Calculates the Relative Height Metrics Correlation between the simulated and original footprints.
        Matching is calculated with:
        - All RH columns Correlation at heights ['25', '50', '60', '70', '75', '80', '85', '90', '95', '98', '100']
          using Pearsonr
        """
        ## RH Correlation
        original_rh_cols = [col for col in original.columns if 'rh_' in col]
        clean_original_rh_cols = clean_cols_rh(original_rh_cols, original=False)
        original_rh = [original[x].values[0] for x in original.columns if x in clean_original_rh_cols]

        simulated_rh_columns = [x for x in simulated.columns if 'rhGauss_' in x]

        correls = []
        # Calculate correlation of RHs between original and simulated
        for i, sim in simulated.iterrows():
            simulated_rh = [sim[x] for x in simulated_rh_columns]

            r = pearsonr(original_rh, simulated_rh)
            correls.append(r.statistic)

        simulated['rh_correlation'] = correls
        
        return simulated

    def _rh_distance(self, original, simulated):
        """
        Calculates the Relative Height Metrics between the simulated and original footprints.
        Matching is calculated with:
        - All RH columns distance at heights ['25', '50', '60', '70', '75', '80', '85', '90', '95', '98', '100']
          using sqrt( sum( (real_x, sim_x)**2 ) )
        """
        rh_col_types = [f'{i}' for i in range(25, 105, 5)]

        original_rh_list = [original[f'rh_{num}'].values[0] for num in rh_col_types]

        rh_distances = []

        for i, row in simulated.iterrows():

            sim_rh_list = [row[f'rhGauss_{num}'] for num in rh_col_types]

            # CRSSDA to RH Profile
            rh_distance_value = CRSSDA(sim_rh_list, original_rh_list)
            rh_distances.append(rh_distance_value)
        
        simulated['rh_distance'] = rh_distances

        return simulated

    def _terrain_matching(self, original, simulated):
        """
        Calculates the difference between each calculated ground elevation from the simulated and original footprints
        Checks which 'ground' or 'product' variable to calculate the difference from at class creation
        """

        if self.ground == 'GEDI':
            # Terrain Matching L2A_1 : GEDI L2A Lowest_Mode - simulated ZG
            simulated_terrain = simulated['ZG']

        if self.ground == 'ALS':
            # Terrain Matching L2A_2 : GEDI L2A Lowest_Mode - true_ground (ALS)
            simulated_terrain = simulated['true_ground']

        simulated['terrain_matching'] = AGED(original['elev_lowestmode'].values, simulated_terrain)

        if self.add_info:
            simulated['elev_lowestmode'] = original['elev_lowestmode'].values[0]
            simulated['digital_elev'] = original['digital_elevation_model'].values[0]
        
        return simulated

    def _waveform_matching(self, original, simulated):
        """
        """

        original_rxwaveform = [float(x) for x in original['rxwaveform'].values[0].split(",")]
        original_binzero = original['geolocation_elevation_bin0'].values[0]
        original_lastbin = original['geolocation_elevation_lastbin'].values[0]
        original_nbins = original['rx_sample_count'].values[0]

        # Build Z-array for original waveform
        gedi_z = np.linspace(original_binzero, original_lastbin, num=original_nbins)

        correlations = []
        wave_distances = []
        kl_scores = []

        for i, sim_point in simulated.iterrows():

            # Build Z-array for simulated waveform
            sim_z = np.linspace(sim_point['Z0'], sim_point['ZN'], num=sim_point['NBINS'])
            
            # Adjust and parse original and simulated waveforms
            original_rxwaveform, sim_rxwaveform = adjust_waveforms(original_rxwaveform, sim_point['RXWAVECOUNT'], nbins=original_nbins)

            # Interpolate both waveforms on a common Z
            ori_wave_interp, sim_wave_interp = interpolate_waveforms(original_rxwaveform, gedi_z, sim_rxwaveform, sim_z)

            # Find bounds of both waveforms. Normally the bounds are when the simulated is different than 0
            startsim, endsim = find_simulated_waveform_bounds(sim_wave_interp, threshold=6.68*0.0001)

            # Processed waveforms
            processed_original_waveform = ori_wave_interp[startsim:endsim]
            processed_simulated_waveform = sim_wave_interp[startsim:endsim]

            # Correlation
            if "wave_pearson" in self.criteria:
                pearson_r = pearson_correlation(ori_wave_interp[startsim:endsim], sim_wave_interp[startsim:endsim])
                correlations.append(pearson_r)
            
            if "wave_spearman" in self.criteria:
                spearman_r = spearman_correlation(ori_wave_interp[startsim:endsim], sim_wave_interp[startsim:endsim])
                correlations.append(spearman_r)

            # Waveform distance
            if "wave_distance" in self.criteria:
                wave_distance_value = CRSSDA(ori_wave_interp[startsim:endsim], sim_wave_interp[startsim:endsim])
                wave_distances.append(wave_distance_value)

            # KL Divergence
            if "kl" in self.criteria:
                kl_value = KL(ori_wave_interp[startsim:endsim], sim_wave_interp[startsim:endsim])
                kl_scores.append(kl_value)

            # Used for debugging plots
            if self.debug:

                footprint_shot_number = original['shot_number_x'].values[0]
                #offset = str(sim_point['grid_offset'])
                
                if footprint_shot_number in []:
                    common_z = np.sort(np.unique(np.concatenate((sim_z, gedi_z))))
                    plot_waveform_comparison(sim_wave_interp,
                                            ori_wave_interp,
                                            common_z,
                                            common_z,
                                            f"{original['shot_number_x'].values[0]}-ID_{sim_point['wave_ID']}",
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0)
                

        if correlations:
            simulated['waveform_matching'] = correlations
        
        if wave_distances:
            simulated['waveform_distance'] = wave_distances

        if kl_scores:
            simulated['kl_distance'] = kl_scores

        if self.add_info:
            #simulated['mean_correl_waveform'] = simulated['waveform_matching'].mean() # Calculate mean waveform matching correlation
            simulated['mean_correl_waveform'] = 0

        return simulated


    def _calculate_score(self, footprint):
        """
        Calculates a final score between 0 and 1 of all selected criteria.
        Before adding to final score, the "_matching" or "_correlation" criteria are normalized between 0 and 1.
        """
        add_score = 0

        for criterion in self.criteria:
            if criterion == 'wave_pearson' or criterion == 'wave_spearman':
                add_score += self._normalize_correl(footprint['waveform_matching'])

            if criterion == 'rh_distance':
                add_score += self._normalize(footprint['rh_distance'].abs())

            if criterion == 'wave_distance':
                add_score += self._normalize(footprint['waveform_distance'])

            if criterion == "kl":
                add_score += self._normalize(footprint['kl_distance'])

            if criterion == 'terrain':
                add_score += self._normalize(footprint['terrain_matching'].abs())

        scores = add_score / len(self.criteria)  # Final Score calculation

        return scores

    def _normalize(self, column):
        """
        Normalizes column in abs()
        """
        return 1 - ((column - column.min()) / (column.max() - column.min()))
        
    def _normalize_correl(self, column):
        """
        Normalizes column of correlation
        """
        return (column - column.min()) / (column.max() - column.min())