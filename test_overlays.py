#!/usr/bin/env python3
"""Test Classical Overlays detection."""
import sys
import numpy as np
sys.path.insert(0, '.')

from genome_sonics import classical_overlays

def test_coding():
    print("Testing Coding Regions...")
    # ATG ... TAA
    seq = "CCCCATGAAAAAATAA"
    #      0123456789012345
    # Coding: 4 to 15 (inclusive start, exclusive end? No, explicit mask)
    mask = classical_overlays.detect_coding_regions(seq)
    print(f"Mask: {mask.astype(int)}")
    assert mask[4], "ATG start should be True"
    assert mask[8], "Middle should be True"
    assert mask[13], "TAA should be True" 
    assert not mask[0], "Pre-coding should be False"
    print("OK")

def test_repeats():
    print("Testing Repeats...")
    # Homopolymer AAAAA (5) at 0
    # STR ATATAT (ATx3) at 10
    seq = "AAAAAGGGGGATATAT"
    indices = classical_overlays.detect_repeats(seq, run_threshold=5, min_copies=3)
    print(f"Indices: {indices}")
    assert 0 in indices, "Should detect AAAAA at 0"
    assert 5 in indices, "Should detect GGGGG at 5"
    assert 10 in indices, "Should detect ATATAT at 10"
    print("OK")

def test_cpg():
    print("Testing CpG...")
    seq = "ACGTACGT"
    #      01234567
    # CpG at 1 (CG) and 5 (CG)
    indices = classical_overlays.detect_cpg_sites(seq)
    print(f"CpG Indices: {indices}")
    assert 1 in indices
    assert 5 in indices
    print("OK")
    
if __name__ == "__main__":
    test_coding()
    test_repeats()
    test_cpg()
