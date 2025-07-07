"""
Helper functions that calculate metrics for waveform similarity between original and simulated
"""

from .waveform_processing import *
from scipy.stats import pearsonr, spearmanr, entropy, linregress
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import math

def pearson_correlation(original_wave, simulated_wave):
    """
    Calculates the Pearson Correlation metric.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Args:
        original_wave (array_like): Original waveform of reported footprint.
        simulated_wave (array_like): Simulated waveform for reported footprint.

    Returns:
        float: correlation r value between original and simulated waveforms.
    """
    r = pearsonr(original_wave, simulated_wave)
    return round(r.statistic, 5)


def spearman_correlation(original_wave, simulated_wave):
    """
    Calculates the Spearman Correlation metric.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

    Args:
        original_wave (array_like): Original waveform of reported footprint.
        simulated_wave (array_like): Simulated waveform for reported footprint.

    Returns:
        float: correlation r value between original and simulated waveforms.
    """
    r = spearmanr(original_wave, simulated_wave)
    return round(r.statistic, 5)


def CRSSDA(original_array, simulated_array):
    """
    Calculates the Curve Root Sum Squared Differential Area
    for waveforms or RH profile.

    Args:
        original_array (array_like): Original array (waveform or RH profile) of reported footprint.
        simulated_array (array_like): Simulated array (waveform or RH profile) of values for reported footprint.

    Returns:
        float: Result of CRSSDA. Rounded to 5 decimal places.
    """
    sub = [(real - sim)**2 for sim, real in zip(simulated_array, original_array)]
    alignment = sum(sub)
    return round(math.sqrt(alignment), 5)


def KL(original_wave, simulated_wave):
    """
    Calculates the Kullback-Leibler Divergence between
    two probabilistic functions using entropy().

    Args:
        original_wave (array_like): Original waveform of reported footprint.
        simulated_wave (array_like): Simulated waveform for reported footprint.

    Returns:
        kl_score (float): KL Score. Rounded to 5 decimal places.
    """
    # Normalize both waveforms from 0 to 1
    normed_ori_wave = normalize_waveform(original_wave) 
    normed_sim_wave = normalize_waveform(simulated_wave)

    original_wave = np.where(normed_ori_wave <= 0, 0.0000000001, normed_ori_wave)
    sim_wave = np.where(normed_sim_wave <= 0, 0.0000000001, normed_sim_wave)

    kl_score = entropy(original_wave, sim_wave)

    # Return rounded kl score to 5 decimal places
    return round(kl_score, 5)


def AGED(original, simulated):
    """
    Calculates the Aboveground Elevation Difference (AGED) between 
    original and simulated terrain elevation estimates.

    Args:
        original (float): Original terrain elevation of reported footprint.
        simulated (float): Simulated terrain elevation for reported footprint.

    Returns:
        float: AGED score.
    """
    return (original - simulated).abs()