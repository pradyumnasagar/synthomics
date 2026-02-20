#!/usr/bin/env python3
"""Test Advanced Sargam Quantization."""
import sys
import numpy as np
sys.path.insert(0, '.')

from genome_sonics import quantize, music

def test_sargam_quantizer():
    print("=== Testing SargamQuantizer ===")
    q = quantize.SargamQuantizer(sa_hz=261.63)
    
    # Test 1: Nearest Neighbor
    input_freq = 300.0 
    quant_nearest = q.quantize(input_freq, mode='nearest')
    print(f"Input: {input_freq} Hz -> Nearest: {quant_nearest:.2f} Hz")
    
    # Test 2: Soft Quantization
    quant_soft = q.quantize(input_freq, mode='soft', strength=0.5)
    print(f"Input: {input_freq} Hz -> Soft (0.5): {quant_soft:.2f} Hz")
    assert quant_nearest != quant_soft
    
    # Test 3: Smart Quantization (Smoothness)
    print("\nTest 3: Smart Quantization (Continuity)")
    q.reset()
    # Create a sequence that sweeps up
    freqs = np.linspace(261.63, 523.25, 10)
    
    print("Nearest Path:")
    prev_n = None
    jumps_n = 0
    for f in freqs:
        val = q.quantize(f, mode='nearest')
        print(f"  {f:.1f} -> {val:.1f}")
        if prev_n and abs(val - prev_n) > 50: jumps_n += 1
        prev_n = val
        
    print("\nSmart Path:")
    q.reset()
    prev_s = None
    jumps_s = 0
    for f in freqs:
        val = q.quantize(f, mode='smart', smoothness=0.9)
        print(f"  {f:.1f} -> {val:.1f}")
        prev_s = val

def test_music_integration():
    print("\n=== Testing Music Integration ===")
    seq = "ATGC"
    
    # Policy 1: Fundamental Only
    print("Policy: Fundamental Only")
    audio, notes = music.generate_music(
        seq,
        scale_type='thz_physics',
        thz_quantize=True,
        quant_policy='fundamental_only'
    )
    partials = notes[0]['partials']
    p0 = partials[0]
    p1 = partials[1]
    ratio = p1/p0
    print(f"  P0: {p0:.2f}, P1: {p1:.2f}, Ratio: {ratio:.3f}")
    
    # Policy 2: All
    print("Policy: All")
    audio2, notes2 = music.generate_music(
        seq,
        scale_type='thz_physics',
        thz_quantize=True,
        quant_policy='all'
    )
    partials2 = notes2[0]['partials']
    p0_2 = partials2[0]
    p1_2 = partials2[1]
    ratio2 = p1_2/p0_2
    print(f"  P0: {p0_2:.2f}, P1: {p1_2:.2f}, Ratio: {ratio2:.3f}")

if __name__ == "__main__":
    test_sargam_quantizer()
    test_music_integration()
    print("\n✅ Verification Complete")
