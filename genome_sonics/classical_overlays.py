"""
Classical Mode Overlays.

This module detects genomic features for structural sonification overlays:
- Coding Regions (ORF) -> Pa
- Repeats/Homopolymers -> Dha
- High Entropy -> Ni
- CpG Sites -> Ornament
"""

import numpy as np
from typing import List, Tuple, Dict
from . import stats

def detect_coding_regions(seq: str) -> np.ndarray:
    """
    Detect simple Coding Regions (ORF-like).
    Region starts at ATG and ends at TAA, TAG, or TGA.
    
    Args:
        seq: DNA sequence
    
    Returns:
        Boolean mask array (True = inside coding region)
    """
    seq = seq.upper()
    mask = np.zeros(len(seq), dtype=bool)
    n = len(seq)
    
    i = 0
    while i < n - 2:
        # Find Start Codon
        if seq[i:i+3] == 'ATG':
            start = i
            # Find Stop Codon (in frame)
            found_stop = False
            for j in range(i + 3, n - 2, 3):
                codon = seq[j:j+3]
                if codon in ['TAA', 'TAG', 'TGA']:
                    stop = j + 3
                    # Mark region (inclusive of start and stop codons)
                    mask[start:stop] = True
                    i = stop
                    found_stop = True
                    break
            
            if not found_stop:
                # No stop found, skip this ATG
                i += 1
        else:
            i += 1
            
    return mask

def detect_repeats(seq: str, 
                   run_threshold: int = 5,
                   min_motif: int = 2,
                   max_motif: int = 6,
                   min_copies: int = 3) -> List[int]:
    """
    Detect start indices of:
    1. Homopolymer runs (e.g., AAAAA) >= run_threshold
    2. Short tandem repeats (motifs 2-6bp) >= min_copies
    
    Args:
        seq: DNA sequence
        run_threshold: Min length for homopolymers
        min_motif: Min motif length for STRs
        max_motif: Max motif length for STRs
        min_copies: Min copies for STRs
        
    Returns:
        List of start indices (unique, sorted)
    """
    seq = seq.upper()
    indices = set()
    n = len(seq)
    
    # 1. Homopolymers
    i = 0
    while i < n:
        current_base = seq[i]
        length = 1
        j = i + 1
        while j < n and seq[j] == current_base:
            length += 1
            j += 1
        
        if length >= run_threshold:
            indices.add(i)
        
        i = j
        
    # 2. STRs (using stats module logic but returning indices)
    # Note: stats.detect_tandem_repeats might be slow for very long seqs if not optimized,
    # but for typical demo seqs it's fine.
    
    # We implement a custom search here to ensure we capture exactly what we want
    # and to be efficient.
    
    visited = set() # To avoid marking same region multiple times
    
    for motif_len in range(min_motif, max_motif + 1):
        i = 0
        while i <= n - motif_len * min_copies:
            if i in visited:
                i += 1
                continue
                
            motif = seq[i:i+motif_len]
            if 'N' in motif:
                i += 1
                continue
                
            # Count copies
            copies = 1
            pos = i + motif_len
            while pos + motif_len <= n and seq[pos:pos+motif_len] == motif:
                copies += 1
                pos += motif_len
                
            if copies >= min_copies:
                indices.add(i)
                # Mark region as visited to avoid sub-motif detection (heuristic)
                for k in range(i, pos):
                    visited.add(k)
                i = pos
            else:
                i += 1
                
    return sorted(list(indices))

def detect_high_entropy(seq: str, 
                        window: int = 64, 
                        percentile: int = 85) -> np.ndarray:
    """
    Detect high entropy regions.
    
    Args:
        seq: DNA sequence
        window: Sliding window size
        percentile: Percentile threshold (0-100)
        
    Returns:
        Boolean mask array
    """
    # Use stats module efficiently
    if len(seq) < window:
        return np.zeros(len(seq), dtype=bool)
        
    local_entropy = stats.calculate_local_entropy(seq, window_size=max(1, int(window/4))) 
    # Note: stats.calculate_local_entropy steps by window_size in the current implementation.
    # To get per-base resolution or finer resolution, we might need a custom function.
    # The current stats.calculate_local_entropy hops by window_size.
    # We want a smoother mask. Let's implement a sliding window here with step=1 or step=4.
    
    # Calculating entropy at every base is expensive (O(N*W)). 
    # Optimization: Step size = 4
    step = 4
    mask = np.zeros(len(seq), dtype=bool)
    
    entropy_vals = []
    indices = []
    
    for i in range(0, len(seq) - window, step):
        sub = seq[i:i+window]
        e = stats.calculate_entropy(sub)
        entropy_vals.append(e)
        indices.append(i)
        
    if not entropy_vals:
        return mask
        
    # Determine threshold
    threshold = np.percentile(entropy_vals, percentile)
    
    for idx, e in zip(indices, entropy_vals):
        if e >= threshold:
            mask[idx : idx + step] = True # Mark the step
            # Ideally mark the whole window? 
            # "When position is inside high-entropy window"
            # Let's mark the center of the window? 
            # Or simpler: just mark the segment covered by this step.
            
    return mask

def detect_cpg_sites(seq: str) -> List[int]:
    """
    Detect CpG sites ('CG' dinucleotides).
    
    Returns:
        List of start indices (indices of 'C')
    """
    seq = seq.upper()
    indices = []
    for i in range(len(seq) - 1):
        if seq[i:i+2] == 'CG':
            indices.append(i)
    return indices
