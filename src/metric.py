"""
Helper functions that calculate metrics for waveform similarity between original and simulated
"""

from .waveform_processing import *
from scipy.stats import pearsonr, spearmanr, norm, ecdf

def pearson_correlation(original_wave, simulated_wave):
    """
    Calculates the Pearson Correlation metric
    """
    r = pearsonr(original_wave, simulated_wave)
    return r.statistic


def spearman_correlation(original_wave, simulated_wave):
    """
    Calculates the Spearman Correlation metric
    """
    r = spearmanr(original_wave, simulated_wave)
    return r.statistic


def CRSSDA(original_array, simulated_array):
    """
    Calculates the Curve Root Sum Squared Differential Area
    for waveforms or RH profile
    """
    sub = [(real - sim)**2 for sim, real in zip(simulated_array, original_array)]
    alignment = sum(sub)
    return math.sqrt(alignment)


def KL(original_wave, simulated_wave):
    """
    Calculates the Kullback-Leibler Divergence
    """
    # Normalize both waveforms from 0 to 1
    normed_ori_wave = normalize_waveform(original_wave) 
    normed_sim_wave = normalize_waveform(simulated_wave)

    kl_score = kl_divergence(normed_ori_wave, normed_sim_wave)
    return kl_score


def AGED(original, simulated):
    """
    Calculates the Aboveground Elevation Difference (AGED) between 
    original and simulated terrain elevation estimates
    """
    return (original - simulated).abs()