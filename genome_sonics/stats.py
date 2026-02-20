"""
Stats Module - Sequence statistics and analysis.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


def calculate_gc_content(seq: str) -> float:
    """
    Calculate GC content percentage.
    
    Args:
        seq: DNA sequence (uppercase)
        
    Returns:
        GC percentage (0-100)
    """
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    valid_count = len([c for c in seq if c in 'ATGC'])
    
    if valid_count == 0:
        return 0.0
    
    return (gc_count / valid_count) * 100


def calculate_entropy(seq: str) -> float:
    """
    Calculate Shannon entropy of the sequence.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Entropy in bits (0 to 2 for DNA)
    """
    seq = seq.upper()
    # Filter to valid bases
    valid_seq = [c for c in seq if c in 'ATGC']
    
    if len(valid_seq) == 0:
        return 0.0
    
    counts = Counter(valid_seq)
    total = len(valid_seq)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def calculate_n_percent(seq: str) -> float:
    """
    Calculate percentage of N (unknown) bases.
    
    Args:
        seq: DNA sequence
        
    Returns:
        N percentage (0-100)
    """
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    
    n_count = seq.count('N')
    return (n_count / len(seq)) * 100


def get_base_frequencies(seq: str) -> Dict[str, float]:
    """
    Calculate frequency of each base.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Dict with base frequencies
    """
    seq = seq.upper()
    total = len([c for c in seq if c in 'ATGCN'])
    
    if total == 0:
        return {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    
    counts = Counter(c for c in seq if c in 'ATGCN')
    return {base: counts.get(base, 0) / total for base in 'ATGCN'}


def get_kmer_frequencies(seq: str, k: int = 3) -> Dict[str, int]:
    """
    Calculate k-mer frequencies.
    
    Args:
        seq: DNA sequence
        k: k-mer size (default 3 for codons)
        
    Returns:
        Dict of k-mer counts
    """
    seq = seq.upper()
    kmers = Counter()
    
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(c in 'ATGC' for c in kmer):
            kmers[kmer] += 1
    
    return dict(kmers)


def get_kmer_vector(seq: str, k: int = 3) -> np.ndarray:
    """
    Get k-mer frequency as a normalized vector.
    
    Args:
        seq: DNA sequence
        k: k-mer size
        
    Returns:
        Normalized frequency vector
    """
    # Generate all possible k-mers
    bases = 'ATGC'
    all_kmers = []
    
    def generate_kmers(prefix, k):
        if k == 0:
            all_kmers.append(prefix)
            return
        for base in bases:
            generate_kmers(prefix + base, k - 1)
    
    generate_kmers('', k)
    
    # Count k-mers in sequence
    freq = get_kmer_frequencies(seq, k)
    
    # Create vector
    vector = np.array([freq.get(kmer, 0) for kmer in all_kmers], dtype=float)
    
    # Normalize
    total = vector.sum()
    if total > 0:
        vector = vector / total
    
    return vector


def get_codon_usage(seq: str) -> Dict[str, int]:
    """
    Calculate codon usage (reading frame 1).
    
    Args:
        seq: DNA sequence
        
    Returns:
        Dict of codon counts
    """
    seq = seq.upper()
    codons = Counter()
    
    # Read in frame
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) == 3 and all(c in 'ATGC' for c in codon):
            codons[codon] += 1
    
    return dict(codons)


def calculate_local_gc(seq: str, window_size: int = 64) -> List[float]:
    """
    Calculate GC content in sliding windows.
    
    Args:
        seq: DNA sequence
        window_size: Window size
        
    Returns:
        List of GC percentages per window
    """
    seq = seq.upper()
    gc_values = []
    
    for i in range(0, len(seq), window_size):
        window = seq[i:i + window_size]
        gc_values.append(calculate_gc_content(window))
    
    return gc_values


def calculate_local_entropy(seq: str, window_size: int = 64) -> List[float]:
    """
    Calculate entropy in sliding windows.
    
    Args:
        seq: DNA sequence
        window_size: Window size
        
    Returns:
        List of entropy values per window
    """
    seq = seq.upper()
    entropy_values = []
    
    for i in range(0, len(seq), window_size):
        window = seq[i:i + window_size]
        entropy_values.append(calculate_entropy(window))
    
    return entropy_values


def detect_tandem_repeats(seq: str, min_unit: int = 2, max_unit: int = 6, 
                          min_copies: int = 3) -> List[Dict]:
    """
    Detect tandem repeats in sequence.
    
    Args:
        seq: DNA sequence
        min_unit: Minimum repeat unit length
        max_unit: Maximum repeat unit length
        min_copies: Minimum number of copies
        
    Returns:
        List of repeat dicts with 'unit', 'copies', 'start', 'end'
    """
    seq = seq.upper()
    repeats = []
    
    for unit_len in range(min_unit, max_unit + 1):
        i = 0
        while i < len(seq) - unit_len * min_copies:
            unit = seq[i:i + unit_len]
            
            if 'N' in unit:
                i += 1
                continue
            
            # Count consecutive copies
            copies = 1
            pos = i + unit_len
            while pos + unit_len <= len(seq) and seq[pos:pos + unit_len] == unit:
                copies += 1
                pos += unit_len
            
            if copies >= min_copies:
                repeats.append({
                    'unit': unit,
                    'copies': copies,
                    'start': i,
                    'end': pos,
                    'length': pos - i
                })
                i = pos  # Skip past this repeat
            else:
                i += 1
    
    return repeats


def get_repeat_density(seq: str) -> float:
    """
    Calculate what fraction of the sequence is tandem repeats.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Repeat density (0-1)
    """
    repeats = detect_tandem_repeats(seq)
    
    if len(seq) == 0:
        return 0.0
    
    # Calculate total repeat length (avoiding overlaps)
    covered = set()
    for r in repeats:
        for pos in range(r['start'], r['end']):
            covered.add(pos)
    
    return len(covered) / len(seq)


def get_sequence_stats(seq: str) -> Dict:
    """
    Get all sequence statistics in one call.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Dict with all statistics
    """
    seq = seq.upper()
    
    # Basic stats
    length = len(seq)
    gc = calculate_gc_content(seq)
    entropy = calculate_entropy(seq)
    n_percent = calculate_n_percent(seq)
    
    # Base frequencies
    base_freq = get_base_frequencies(seq)
    
    # Codon usage (top 10)
    codon_usage = get_codon_usage(seq)
    top_codons = sorted(codon_usage.items(), key=lambda x: -x[1])[:10]
    
    # Repeats
    repeats = detect_tandem_repeats(seq)
    repeat_density = get_repeat_density(seq)
    
    # Local variation
    local_gc = calculate_local_gc(seq)
    local_entropy = calculate_local_entropy(seq)
    
    return {
        'length': length,
        'gc_content': round(gc, 2),
        'entropy': round(entropy, 4),
        'n_percent': round(n_percent, 2),
        'base_frequencies': {k: round(v, 4) for k, v in base_freq.items()},
        'top_codons': top_codons,
        'repeat_count': len(repeats),
        'repeat_density': round(repeat_density, 4),
        'gc_mean': round(np.mean(local_gc), 2) if local_gc else 0,
        'gc_std': round(np.std(local_gc), 2) if local_gc else 0,
        'entropy_mean': round(np.mean(local_entropy), 4) if local_entropy else 0,
        'entropy_std': round(np.std(local_entropy), 4) if local_entropy else 0,
    }


def format_stats_for_display(stats: Dict) -> str:
    """Format stats dict as readable text."""
    lines = [
        f"**Length:** {stats['length']:,} bp",
        f"**GC Content:** {stats['gc_content']:.1f}%",
        f"**Shannon Entropy:** {stats['entropy']:.3f} bits",
        f"**N (Unknown):** {stats['n_percent']:.1f}%",
        "",
        "**Base Frequencies:**",
    ]
    
    for base in 'ATGC':
        freq = stats['base_frequencies'].get(base, 0)
        lines.append(f"  {base}: {freq*100:.1f}%")
    
    lines.extend([
        "",
        f"**Tandem Repeats:** {stats['repeat_count']} found",
        f"**Repeat Density:** {stats['repeat_density']*100:.1f}%",
        "",
        f"**GC Variation:** μ={stats['gc_mean']:.1f}%, σ={stats['gc_std']:.1f}%",
        f"**Entropy Variation:** μ={stats['entropy_mean']:.3f}, σ={stats['entropy_std']:.3f}",
    ])
    
    return '\n'.join(lines)
