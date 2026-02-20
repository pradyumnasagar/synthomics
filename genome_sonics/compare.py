"""
Compare Module - Sequence comparison and similarity metrics.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter


def kmer_vector(seq: str, k: int = 3) -> np.ndarray:
    """
    Create normalized k-mer frequency vector.
    
    Args:
        seq: DNA sequence
        k: k-mer length
        
    Returns:
        Normalized frequency vector
    """
    from . import stats
    return stats.get_kmer_vector(seq, k)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Similarity score (0-1)
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


def jensen_shannon_divergence(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        v1: First probability vector (will be normalized)
        v2: Second probability vector (will be normalized)
        
    Returns:
        JSD value (0 = identical, 1 = maximally different)
    """
    # Normalize to probability distributions
    p = v1 / (v1.sum() + 1e-10)
    q = v2 / (v2.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    # Compute mixture
    m = 0.5 * (p + q)
    
    # KL divergences
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    
    # JSD
    jsd = 0.5 * (kl_pm + kl_qm)
    
    return float(np.clip(jsd, 0, 1))


def edit_distance(s1: str, s2: str, max_len: int = 1000) -> Optional[int]:
    """
    Compute Levenshtein edit distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
        max_len: Maximum length to compute (for performance)
        
    Returns:
        Edit distance, or None if sequences too long
    """
    if len(s1) > max_len or len(s2) > max_len:
        return None
    
    m, n = len(s1), len(s2)
    
    # Use only two rows for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    
    return prev[n]


def normalized_edit_distance(s1: str, s2: str, max_len: int = 1000) -> Optional[float]:
    """
    Compute normalized edit distance (0-1 scale).
    
    Args:
        s1: First sequence
        s2: Second sequence
        max_len: Maximum length
        
    Returns:
        Normalized distance (0 = identical, 1 = completely different)
    """
    ed = edit_distance(s1, s2, max_len)
    if ed is None:
        return None
    
    max_possible = max(len(s1), len(s2))
    if max_possible == 0:
        return 0.0
    
    return ed / max_possible


def base_composition_similarity(seq1: str, seq2: str) -> float:
    """
    Compare base composition between sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Similarity score (0-1)
    """
    from . import stats
    
    freq1 = stats.get_base_frequencies(seq1)
    freq2 = stats.get_base_frequencies(seq2)
    
    # Euclidean distance in frequency space
    dist = 0.0
    for base in 'ATGC':
        dist += (freq1.get(base, 0) - freq2.get(base, 0)) ** 2
    
    dist = math.sqrt(dist)
    
    # Convert to similarity (max distance = 2 for opposite compositions)
    similarity = 1.0 - (dist / 2.0)
    
    return max(0.0, similarity)


def compare_sequences(seq1: str, seq2: str,
                      name1: str = "Sequence 1",
                      name2: str = "Sequence 2",
                      k: int = 3) -> Dict:
    """
    Comprehensive comparison between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        name1: Name for first sequence
        name2: Name for second sequence
        k: k-mer size for vectorization
        
    Returns:
        Dict with comparison metrics and stats
    """
    from . import stats
    
    # Get stats for each sequence
    stats1 = stats.get_sequence_stats(seq1)
    stats2 = stats.get_sequence_stats(seq2)
    
    # K-mer vectors
    vec1 = kmer_vector(seq1, k)
    vec2 = kmer_vector(seq2, k)
    
    # Similarity metrics
    cos_sim = cosine_similarity(vec1, vec2)
    jsd = jensen_shannon_divergence(vec1, vec2)
    base_sim = base_composition_similarity(seq1, seq2)
    
    # Edit distance (only for short sequences)
    ed = None
    ned = None
    if len(seq1) <= 1000 and len(seq2) <= 1000:
        ed = edit_distance(seq1, seq2)
        ned = normalized_edit_distance(seq1, seq2)
    
    return {
        'sequence1': {
            'name': name1,
            'stats': stats1
        },
        'sequence2': {
            'name': name2,
            'stats': stats2
        },
        'similarity': {
            'cosine_similarity': round(cos_sim, 4),
            'jensen_shannon_divergence': round(jsd, 4),
            'base_composition_similarity': round(base_sim, 4),
            'edit_distance': ed,
            'normalized_edit_distance': round(ned, 4) if ned else None,
        },
        'interpretation': _interpret_similarity(cos_sim, jsd)
    }


def _interpret_similarity(cos_sim: float, jsd: float) -> str:
    """Generate human-readable interpretation of similarity metrics."""
    
    if cos_sim > 0.95:
        level = "Nearly identical"
        detail = "These sequences have almost the same compositional signature."
    elif cos_sim > 0.8:
        level = "Highly similar"
        detail = "These sequences share strong compositional patterns."
    elif cos_sim > 0.6:
        level = "Moderately similar"
        detail = "These sequences have some shared patterns but notable differences."
    elif cos_sim > 0.4:
        level = "Somewhat different"
        detail = "These sequences have different compositional profiles."
    else:
        level = "Very different"
        detail = "These sequences have highly distinct compositional signatures."
    
    jsd_note = ""
    if jsd < 0.1:
        jsd_note = "Information content is virtually identical."
    elif jsd < 0.3:
        jsd_note = "Information content is similar."
    else:
        jsd_note = "Information content differs significantly."
    
    return f"**{level}**: {detail} {jsd_note}"


def format_comparison_report(comparison: Dict) -> str:
    """Format comparison results as readable markdown."""
    
    s1 = comparison['sequence1']
    s2 = comparison['sequence2']
    sim = comparison['similarity']
    
    lines = [
        f"# Comparison: {s1['name']} vs {s2['name']}",
        "",
        "## Overview",
        f"| Metric | {s1['name']} | {s2['name']} |",
        "|--------|------------|------------|",
        f"| Length | {s1['stats']['length']:,} bp | {s2['stats']['length']:,} bp |",
        f"| GC Content | {s1['stats']['gc_content']:.1f}% | {s2['stats']['gc_content']:.1f}% |",
        f"| Entropy | {s1['stats']['entropy']:.3f} | {s2['stats']['entropy']:.3f} |",
        "",
        "## Similarity Metrics",
        f"| Metric | Value | Interpretation |",
        f"|--------|-------|----------------|",
        f"| Cosine Similarity | {sim['cosine_similarity']:.4f} | {'High' if sim['cosine_similarity'] > 0.7 else 'Medium' if sim['cosine_similarity'] > 0.4 else 'Low'} |",
        f"| Jensen-Shannon Div. | {sim['jensen_shannon_divergence']:.4f} | {'Similar' if sim['jensen_shannon_divergence'] < 0.3 else 'Different'} |",
        f"| Base Composition | {sim['base_composition_similarity']:.4f} | {'Match' if sim['base_composition_similarity'] > 0.8 else 'Differ'} |",
    ]
    
    if sim['edit_distance'] is not None:
        lines.append(f"| Edit Distance | {sim['edit_distance']} | {sim['normalized_edit_distance']:.2%} different |")
    
    lines.extend([
        "",
        "## Interpretation",
        comparison['interpretation']
    ])
    
    return '\n'.join(lines)


def get_comparison_legend() -> str:
    """Get explanation of comparison metrics."""
    return """
## Comparison Metrics Explained

### Cosine Similarity (0-1)
Measures angle between k-mer frequency vectors.
- **1.0**: Identical compositional profile
- **0.7+**: Highly similar
- **0.4-0.7**: Moderately similar
- **<0.4**: Different sequences

### Jensen-Shannon Divergence (0-1)
Information-theoretic measure of distribution difference.
- **0.0**: Identical distributions
- **<0.3**: Similar information content
- **>0.5**: Significantly different

### Base Composition Similarity
Compares A/T/G/C ratios directly.
- Useful for quick organism-level comparison
- Less sensitive to sequence order

### Edit Distance
Number of insertions/deletions/substitutions needed.
- Only computed for sequences <1000 bp
- Useful for mutation analysis
"""
