#!/usr/bin/env python3
"""Test THz Physics Mode synthesis."""
import sys
sys.path.insert(0, '.')

from genome_sonics import music, thz_data

# Test thz_data module
print("=== Testing thz_data module ===")
print(f"p_min: {thz_data.get_global_pmin()} THz")
print(f"p_max: {thz_data.get_global_pmax()} THz")

for base in 'ATGC':
    peaks = thz_data.select_peaks(base, k=5, mode='lowest')
    print(f"{base} peaks: {peaks}")

print("\nMapping A[0] to audible:")
for alpha in [0.25, 0.35, 0.45]:
    freq = thz_data.thz_to_audible(1.70, alpha=alpha)
    print(f"  alpha={alpha}: {freq:.1f} Hz")

# Test full synthesis
print("\n=== Testing THz Physics Mode Synthesis ===")
seq = "ATGCATGC" * 5

try:
    audio, notes = music.generate_music(
        seq,
        scale_type='thz_physics',
        thz_k=5,
        thz_alpha=0.35,
        thz_quantize=False
    )
    print(f"✓ Generated {len(audio)} samples")
    print(f"✓ Generated {len(notes)} notes")
    print(f"✓ First note partials: {notes[0].get('partials', 'N/A')}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test existing modes still work
print("\n=== Testing Existing Modes (Regression) ===")
for mode in ['sargam', 'major', 'minor']:
    try:
        audio, notes = music.generate_music(seq, scale_type=mode)
        print(f"✓ {mode}: {len(audio)} samples, {len(notes)} notes")
    except Exception as e:
        print(f"✗ {mode} failed: {e}")

print("\n✅ THz Physics Mode tests complete!")
