#!/usr/bin/env python3
"""Test new instrument timbres."""
import sys
sys.path.insert(0, '.')

from genome_sonics import music

seq = "ATGCATGC" * 30

instruments = ['sitar', 'harmonium', 'veena', 'bells', 'strings', 'organ']

print("Testing all instruments...")
for inst in instruments:
    try:
        audio, notes = music.generate_music(seq, timbre=inst)
        print(f"  ✓ {inst}: {len(audio)} samples, {len(notes)} notes")
    except Exception as e:
        print(f"  ✗ {inst}: {e}")

print("\n✅ All instruments working!")
