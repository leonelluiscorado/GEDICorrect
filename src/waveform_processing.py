"""
Helper functions with waveform data processing and normalization functions
Also includes Plotting function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.interpolate import interp1d

def normalize_waveform(wave):
    """
    Normalizes a waveform from 0 to 1.

    Args:
        wave (array-like): Waveform.

    Returns:
        normalized (array-like): Normalized waveform.
    """
    total = np.sum(wave)
    normalized = wave / total
    return normalized


def adjust_waveforms(gedi_wave, sim_wave, nbins):
    """
    Adjusts both GEDI and Simulated waveforms from gediRat.

    Args:
        gedi_wave (list): Original GEDI waveform.
        sim_wave (list): Simulated GEDI waveform.
        nbins (int): Number of bins of both waveforms.

    Returns:
        adjusted_gedi (array-like): Parsed and processed original GEDI waveform.
        adjusted_simulated (array-like): Parsed and processed simulated GEDI waveform.
    """
    gedi_wave = np.array(gedi_wave)
    sim_wave = np.array(sim_wave)

    noise_bins = int(10/0.15)
    mean_noise = np.mean(gedi_wave[:noise_bins])
    stdev = np.std(gedi_wave[:noise_bins])
    threshold = mean_noise + 3.5 * stdev
    total_energy = np.sum(gedi_wave[gedi_wave > threshold] - mean_noise)

    adjusted_gedi = gedi_wave[0:nbins] - mean_noise
    adjusted_simulated = sim_wave * total_energy / np.sum(sim_wave)

    return adjusted_gedi, adjusted_simulated


def interpolate_waveforms(original_wave, original_z, simulated_wave, simulated_z):
    """
    Interpolates waveforms to a common Z (ground). By aligning both waveforms in terms of ground return,
    they become directly comparable, both original waveforms and simulated waveforms.

    Args:
        original_wave (array-like): Original GEDI waveform.
        original_z (array-like): Original GEDI Z-array (elevation_lastbin to elevation_bin0).
        simulated_wave (array-like): Simulated GEDI waveform.
        simulated_z (array-like): Simulated GEDI Z-array (elevation_lastbin to elevation_bin0).

    Returns:
        ori_wave_interp (array-like): Interpolated original GEDI waveform.
        sim_wave_interp (array-like): Interpolated simulated GEDI waveform.
    """
    # Interpolate both waveforms to a common Z
    common_z = np.sort(np.unique(np.concatenate((simulated_z, original_z))))

    interp_simulated = interp1d(simulated_z, simulated_wave, bounds_error=False, fill_value="extrapolate")
    interp_original = interp1d(original_z, original_wave, bounds_error=False, fill_value="extrapolate")

    sim_wave_interp = interp_simulated(common_z)
    ori_wave_interp = interp_original(common_z)

    return ori_wave_interp, sim_wave_interp


def find_simulated_waveform_bounds(waveform, threshold=6.68*0.0001, return_indices=False):
    """
    Returns the start and end bounds of given waveform values that are above a 'threshold'.
    For GEDI, this threshold is 6.68*0.0001 (Hancock et al. 2019).

    Args:
        waveform (array-like): GEDI waveform.
        threshold (float): Limit to check bounds.
        return_indices (bool): Flag to return indices above threshold.

    Returns
        start_bound (int): Index of start of waveform above threshold.
        end_bound (int): Index of end of waveform above threshold.
        indices (list): (return_indices = True) Indices of each element above threshold.
    """
    start_bound = next((i for i, x in enumerate(waveform) if x > threshold), None)
    end_bound = next((i for i, x in enumerate(reversed(waveform)) if x > threshold), None)

    if end_bound is not None:
        end_bound = len(waveform)- 1 - end_bound
    
    # Find the indices of each non-zero element
    if return_indices:
        indices = [i for i, x in enumerate(waveform) if x > threshold]
        return start_bound, end_bound, indices
    
    return start_bound, end_bound


def align_simulated_gedi(original_rxwaveform, sim_rxwaveform, sim_bounds, original_z, sim_z):
    """
    NOT USED.

    Aligns different sized Z arrays between original and simulated waveforms.
    """
    sim_z_bounds = sim_z[sim_bounds[0]:sim_bounds[1]]

    # Align Simulated Z bounds with Reported GEDI Z bounds
    gedi_z_bounds = original_z[(original_z > sim_z_bounds.min()) & (original_z < sim_z_bounds.max())]

    if len(gedi_z_bounds) == 0:
        return [], []

    gedi_arr_start = np.where(original_z == gedi_z_bounds.min())[0][0]
    gedi_arr_end = np.where(original_z == gedi_z_bounds.max())[0][0]
    
    indices_to_add = abs(len(original_rxwaveform[gedi_arr_end:gedi_arr_start]) - len(sim_rxwaveform[sim_bounds[0]:sim_bounds[1]]))

    return original_rxwaveform[gedi_arr_end:gedi_arr_start+indices_to_add], sim_rxwaveform[sim_bounds[0]:sim_bounds[1]]


def plot_waveform_comparison(sim_wave, gedi_wave, sim_z, gedi_z, out_filename):
    """
    Plots the Original and Simulated Waveforms for debugging purposes.
    Outputs at 600 dpi to 'out_filename'.

    Args:
        sim_wave (array-like): Simulated GEDI waveform.
        gedi_wave (array-like): Original GEDI waveform.
        sim_z (array-like): Simulated GEDI Z-array (elevation_lastbin to elevation_bin0).
        gedi_z (array-like): Original GEDI Z-array (elevation_lastbin to elevation_bin0).
        out_filename (str): Plot output filename.

    Returns:
        True: if plotted successfully.

    """

    xmax = max(max(gedi_wave), max(sim_wave)) + 10

    try:
        plt.ioff()
        plt.plot(sim_wave, sim_z, color='#196e27', label="Simulation")
        plt.plot(gedi_wave, gedi_z, color='#e71c7d', label="GEDI")

        plt.legend(loc='lower right')
        plt.xlim(left=-5)
        plt.xlabel('DN')
        plt.ylabel('Height (m)')

        name_file = f"{out_filename}.png"

        plt.savefig(name_file)
        plt.close()
        plt.clf()
        print("Plotted Waveforms to ", name_file)

    except Exception as e: 
        print(e)
        print("A problem occurred while plotting both waveforms.")
        return False

    return True