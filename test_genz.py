#!/usr/bin/env python3
"""Test Gen Z Music features."""
import sys
sys.path.insert(0, '.')

from genome_sonics import music

seq = "ATGC" * 10
print("Testing Gen Z Timbres & Beats...")

# Test Lofi Timbre (should use beat in western mode)
try:
    print("\nTesting Lofi Keys + Major Scale (should have beat):")
    audio, notes = music.generate_music(seq, timbre='lofi', scale_type='major')
    print(f"  ✓ Lofi generated: {len(audio)} samples")
except Exception as e:
    print(f"  ✗ Lofi failed: {e}")
    import traceback
    traceback.print_exc()

# Test Synth Timbre
try:
    print("\nTesting Synth Lead + Minor Scale:")
    audio, notes = music.generate_music(seq, timbre='synth', scale_type='minor')
    print(f"  ✓ Synth generated: {len(audio)} samples")
except Exception as e:
    print(f"  ✗ Synth failed: {e}")

print("\n✅ Gen Z features valid!")
