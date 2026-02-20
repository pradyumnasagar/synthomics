#!/usr/bin/env python3
"""Quick test script for Genome Sonics modules."""

import sys
sys.path.insert(0, '.')

def test_all():
    print("=" * 50)
    print("Genome Sonics - Module Tests")
    print("=" * 50)
    
    # Test IO
    print("\n[1] Testing IO module...")
    try:
        from genome_sonics import io as gs_io
        demos = gs_io.get_demo_list()
        print(f"    ✓ Found {len(demos)} demo sequences")
        
        result = gs_io.load_demo_sequence('HPV-16 E6/E7')
        if result:
            header, seq = result
            print(f"    ✓ Loaded HPV-16: {len(seq)} bp")
        else:
            print("    ✗ Failed to load demo")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False
    
    # Test Stats
    print("\n[2] Testing Stats module...")
    try:
        from genome_sonics import stats as gs_stats
        s = gs_stats.get_sequence_stats(seq)
        print(f"    ✓ GC: {s['gc_content']:.1f}%")
        print(f"    ✓ Entropy: {s['entropy']:.3f} bits")
        print(f"    ✓ Repeats: {s['repeat_count']}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False
    
    # Test Music
    print("\n[3] Testing Music module...")
    try:
        from genome_sonics import music as gs_music
        audio, notes = gs_music.generate_music(
            seq[:200],
            sa_frequency=261.63,
            timbre='tanpura',
            drone_enabled=True
        )
        print(f"    ✓ Generated {len(audio)} audio samples")
        print(f"    ✓ Generated {len(notes)} notes")
        
        # Check audio range
        import numpy as np
        print(f"    ✓ Audio range: [{audio.min():.2f}, {audio.max():.2f}]")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Art
    print("\n[4] Testing Art module...")
    try:
        from genome_sonics import art as gs_art
        
        for style in ['heatmap_walk', 'chaos', 'mosaic']:
            img = gs_art.generate_art(seq[:200], style=style, width=400, height=400)
            print(f"    ✓ {style}: {img.size}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Compare
    print("\n[5] Testing Compare module...")
    try:
        from genome_sonics import compare as gs_compare
        
        result2 = gs_io.load_demo_sequence('SARS-CoV-2 Spike')
        if result2:
            _, seq2 = result2
            cmp = gs_compare.compare_sequences(seq[:200], seq2[:200])
            print(f"    ✓ Cosine similarity: {cmp['similarity']['cosine_similarity']:.3f}")
            print(f"    ✓ Jensen-Shannon: {cmp['similarity']['jensen_shannon_divergence']:.3f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    return True

if __name__ == '__main__':
    success = test_all()
    sys.exit(0 if success else 1)
