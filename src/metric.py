"""
Helper functions that calculate metrics for waveform similarity between original and simulated
"""

from .waveform_processing import *
from scipy.stats import pearsonr, spearmanr, norm, ecdf
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
    return r.statistic


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
    return r.statistic


def CRSSDA(original_array, simulated_array):
    """
    Calculates the Curve Root Sum Squared Differential Area
    for waveforms or RH profile.

    Args:
        original_array (array_like): Original array (waveform or RH profile) of reported footprint.
        simulated_array (array_like): Simulated array (waveform or RH profile) of values for reported footprint.

    Returns:
        float: Result of CRSSDA.
    """
    sub = [(real - sim)**2 for sim, real in zip(simulated_array, original_array)]
    alignment = sum(sub)
    return math.sqrt(alignment)


def KL(original_wave, simulated_wave):
    """
    Calculates the Kullback-Leibler Divergence between
    two probabilistic functions.

    Args:
        original_wave (array_like): Original waveform of reported footprint.
        simulated_wave (array_like): Simulated waveform for reported footprint.

    Returns:
        kl_score (float): KL Score.
    """
    # Normalize both waveforms from 0 to 1
    normed_ori_wave = normalize_waveform(original_wave) 
    normed_sim_wave = normalize_waveform(simulated_wave)

    kl_score = kl_divergence(normed_ori_wave, normed_sim_wave)
    return kl_score


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