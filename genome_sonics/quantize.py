"""
Sargam Quantization Module.

This module provides advanced quantization strategies for mapping continuous
frequencies (e.g., from THz Physics Mode) to musical scales.

Key Features:
- Multiple quantization types: 'nearest', 'soft', 'smart' (cost-based)
- Continuity-aware 'smart' quantization to prevent jumpy melodies
- Soft quantization blending for microtonal character
- Stateful processing for smoothness across a sequence
"""

import math
from typing import List, Optional, Tuple, Dict

# Scales defined as semitone offsets from Sa (0)
# All scales span one octave [0, 12]
SCALES = {
    'bilawal': [0, 2, 4, 5, 7, 9, 11, 12],  # Major
    'kalyan':  [0, 2, 4, 6, 7, 9, 11, 12],  # Lydian
    'bhairav': [0, 1, 4, 5, 7, 8, 11, 12],  # Double Harmonic
    'kafi':    [0, 2, 3, 5, 7, 9, 10, 12],  # Dorian
    'chromatic': list(range(13))            # All 12 semitones
}

class SargamQuantizer:
    """
    Stateful quantizer for musical sequences.
    Maintains history to support continuity-aware quantization.
    """
    
    def __init__(self, 
                 scale_name: str = 'bilawal', 
                 sa_hz: float = 261.63):
        """
        Initialize quantizer.
        
        Args:
            scale_name: Name of scale from SCALES dict
            sa_hz: Base frequency (Sa)
        """
        self.scale = SCALES.get(scale_name.lower(), SCALES['bilawal'])
        self.sa_hz = sa_hz
        self.prev_semitone = None  # State for continuity
        
    def set_scale(self, scale_name: str):
        self.scale = SCALES.get(scale_name.lower(), SCALES['bilawal'])
        
    def freq_to_semitone(self, freq: float) -> float:
        """Convert Hz to continuous semitone offset from Sa."""
        if freq <= 0: return 0.0
        return 12.0 * math.log2(freq / self.sa_hz)
    
    def semitone_to_freq(self, semitone: float) -> float:
        """Convert semitone offset to Hz."""
        return self.sa_hz * (2.0 ** (semitone / 12.0))
        
    def _get_candidate_targets(self, input_semitone: float, range_octaves: int = 1) -> List[float]:
        """
        Generate candidate snapped semitones near the input.
        Considers targets in current, previous, and next octaves.
        """
        base_octave = int(input_semitone // 12)
        candidates = []
        
        # Check octaves around the input
        for octave in range(base_octave - range_octaves, base_octave + range_octaves + 1):
            for degree in self.scale[:-1]: # Exclude 12 (it's 0 next octave)
                candidates.append(octave * 12 + degree)
                
        return candidates

    def quantize(self, 
                 freq: float, 
                 mode: str = 'nearest', 
                 strength: float = 0.7,
                 smoothness: float = 0.6) -> float:
        """
        Quantize a frequency according to the selected mode.
        
        Args:
            freq: Input frequency in Hz
            mode: 'nearest', 'soft', or 'smart'
            strength: Quantization strength (0.0-1.0) for 'soft' mode
            smoothness: Continuity weight (0.0-1.0) for 'smart' mode
            
        Returns:
            Quantized frequency in Hz
        """
        # Handle silence or invalid input
        if freq <= 10: return freq
        
        input_semitone = self.freq_to_semitone(freq)
        
        # 1. Determine Target Semitone
        if mode == 'smart' and self.prev_semitone is not None:
             target_semitone = self._smart_snap(input_semitone, smoothness)
        else:
             target_semitone = self._nearest_snap(input_semitone)
        
        # Update state
        self.prev_semitone = target_semitone
        
        # 2. Apply Output Logic
        if mode == 'soft':
            # Blend continuous input with snapped target
            # strength 1.0 = hard snap, 0.0 = no quantization
            out_semitone = (1.0 - strength) * input_semitone + strength * target_semitone
            return self.semitone_to_freq(out_semitone)
            
        else:
            # Hard snap (nearest or smart)
            return self.semitone_to_freq(target_semitone)
            
    def _nearest_snap(self, note: float) -> float:
        """Find the absolute nearest scale degree."""
        candidates = self._get_candidate_targets(note)
        return min(candidates, key=lambda x: abs(x - note))
        
    def _smart_snap(self, note: float, smoothness: float) -> float:
        """
        Find target minimizing cost function:
        Cost = Dist(input, target) + w * Dist(prev, target)
        """
        candidates = self._get_candidate_targets(note, range_octaves=1)
        
        # Weights
        # w1 (fidelity) is fixed at 1.0
        # w2 (continuity) scales with smoothness parameter (0->0, 1->2.0)
        w_continuity = smoothness * 2.5
        
        def cost(target):
            fidelity_cost = abs(target - note)
            continuity_cost = abs(target - self.prev_semitone)
            
            # Additional penalty for large octave jumps (> 1.5 octaves)
            jump_penalty = 0
            if abs(target - self.prev_semitone) > 18:
                jump_penalty = 5.0
                
            return fidelity_cost + (w_continuity * continuity_cost) + jump_penalty
            
        return min(candidates, key=cost)

    def reset(self):
        """Reset history state."""
        self.prev_semitone = None
