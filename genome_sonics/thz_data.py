"""
THz Absorption Peak Data for DNA Nucleobases.

This module provides published THz spectroscopy data for sonification.
The peaks represent characteristic vibrational modes of nucleobases
measured via THz time-domain spectroscopy (THz-TDS).

Primary Source:
    Yu, L. et al. (2019). "Terahertz Absorption Spectroscopy of DNA 
    Nucleobases for Plant Genomes Identification." Sensors, 19(5), 1148.
    Table 2: THz-ABCD peak positions.

Foundational Reference:
    Fischer, B.M. et al. (2002). "Far-infrared vibrational modes of DNA 
    components studied by terahertz time-domain spectroscopy." 
    Physics in Medicine and Biology, 47(21), 3807.

DISCLAIMER:
    Molecular vibrations occur at THz frequencies (10^12 Hz), far above
    human hearing (20-20,000 Hz). This module maps peak POSITIONS into
    audible range via ratio-preserving compression for DATA SONIFICATION.
    We are NOT "hearing molecules" - this is frequency mapping.
"""

import math
from typing import List, Optional

# =============================================================================
# THz PEAK DATA (Yu et al. 2019, Table 2)
# =============================================================================

THZ_PEAKS = {
    'A': [1.70, 2.15, 2.50, 3.10, 4.10, 5.60, 6.00, 7.30],  # Adenine
    'G': [2.50, 3.00, 4.30, 4.80, 5.35, 6.30, 7.20, 9.80],  # Guanine
    'C': [1.55, 2.75, 3.40, 4.35, 4.75, 5.95, 6.95],        # Cytosine
    'T': [1.30, 2.25, 2.95, 4.50, 5.10, 6.30, 8.50, 9.60],  # Thymine
}

# Metadata
CITATION_PRIMARY = "Yu et al., Sensors 2019, 19(5):1148"
CITATION_FOUNDATION = "Fischer et al., Phys Med Biol 2002, 47(21):3807"
UNITS = "THz"
DATASET_NAME = "THz-ABCD Peak Positions"

# Bilawal/Major scale semitone offsets for quantization
SARGAM_SEMITONES = [0, 2, 4, 5, 7, 9, 11, 12]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_global_pmin() -> float:
    """
    Return the minimum THz peak across all nucleobases.
    This serves as the reference for ratio-preserving mapping.
    
    Returns:
        1.30 THz (from Thymine)
    """
    return min(min(peaks) for peaks in THZ_PEAKS.values())


def get_global_pmax() -> float:
    """Return the maximum THz peak across all nucleobases."""
    return max(max(peaks) for peaks in THZ_PEAKS.values())


def select_peaks(base: str, k: int = 5, mode: str = 'lowest') -> List[float]:
    """
    Select K peaks from a nucleobase's THz spectrum.
    
    Args:
        base: Nucleobase letter (A, T, G, C)
        k: Number of peaks to select (3-8)
        mode: Selection strategy
            - 'lowest': Take K lowest-frequency peaks (most distinct)
            - 'even': Evenly spaced across spectrum
            - 'all': Return all peaks (ignores k)
    
    Returns:
        List of THz peak frequencies
    """
    peaks = THZ_PEAKS.get(base.upper(), [])
    if not peaks:
        return []
    
    k = min(k, len(peaks))
    
    if mode == 'lowest':
        return sorted(peaks)[:k]
    elif mode == 'even':
        if k >= len(peaks):
            return peaks
        step = len(peaks) / k
        return [peaks[int(i * step)] for i in range(k)]
    elif mode == 'all':
        return peaks
    else:
        return sorted(peaks)[:k]


def thz_to_audible(peak_thz: float, 
                   sa_hz: float = 261.63, 
                   alpha: float = 0.35) -> float:
    """
    Map a THz peak position to an audible frequency.
    
    Uses ratio-preserving compression:
        f_audible = Sa_Hz × (peak / p_min)^alpha
    
    Args:
        peak_thz: THz absorption peak position
        sa_hz: Base frequency (Sa/Root) in Hz
        alpha: Compression exponent (0.2-0.6, default 0.35)
               Lower = more compressed range
               Higher = wider frequency spread
    
    Returns:
        Audible frequency in Hz
    """
    p_min = get_global_pmin()
    ratio = peak_thz / p_min
    return sa_hz * (ratio ** alpha)


def quantize_to_sargam(freq_hz: float, sa_hz: float = 261.63) -> float:
    """
    Snap a frequency to the nearest Sargam/Bilawal scale degree.
    
    Uses semitone offsets: [0, 2, 4, 5, 7, 9, 11, 12]
    (Sa, Re, Ga, Ma, Pa, Dha, Ni, Sa')
    
    Args:
        freq_hz: Input frequency in Hz
        sa_hz: Sa/Root frequency in Hz
    
    Returns:
        Quantized frequency in Hz
    """
    if freq_hz <= 0 or sa_hz <= 0:
        return sa_hz
    
    semitones = 12 * math.log2(freq_hz / sa_hz)
    
    # Handle octave wrapping
    octave = int(semitones // 12)
    semitones_in_octave = semitones % 12
    
    # Find nearest allowed semitone
    nearest = min(SARGAM_SEMITONES[:-1], key=lambda x: abs(x - semitones_in_octave))
    
    # Reconstruct frequency
    total_semitones = octave * 12 + nearest
    return sa_hz * (2 ** (total_semitones / 12))


def get_base_partials(base: str,
                      k: int = 5,
                      mode: str = 'lowest',
                      sa_hz: float = 261.63,
                      alpha: float = 0.35,
                      quantize: bool = False) -> List[float]:
    """
    Get audible partial frequencies for a nucleobase.
    
    This is the main function for THz Physics Mode sonification.
    
    Args:
        base: Nucleobase (A, T, G, C)
        k: Number of partials
        mode: Peak selection mode
        sa_hz: Base frequency
        alpha: Compression exponent
        quantize: If True, snap to Sargam scale
    
    Returns:
        List of audible frequencies in Hz
    """
    peaks = select_peaks(base, k, mode)
    partials = [thz_to_audible(p, sa_hz, alpha) for p in peaks]
    
    if quantize:
        partials = [quantize_to_sargam(f, sa_hz) for f in partials]
    
    return partials


def get_citation_text() -> str:
    """Return formatted citation for UI display."""
    return f"""**Primary Source**: {CITATION_PRIMARY}
**Foundation**: {CITATION_FOUNDATION}
**Data**: {DATASET_NAME} (unit: {UNITS})"""


def get_disclaimer() -> str:
    """Return scientific disclaimer for UI."""
    return """Molecular vibrations occur at THz frequencies (10¹² Hz), far above 
human hearing (20–20,000 Hz). This mode maps *published THz absorption peak 
positions* into the audible range via ratio-preserving compression. We are 
not "hearing molecules"—this is data sonification using spectroscopic 
fingerprints as frequency input."""
