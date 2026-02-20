"""
Music Module - Sargam-based DNA sonification with harmonic synthesis.

Mapping Strategy:
- Nucleotides → Sargam notes with occurrence-toggle (odd=Cycle1, even=Cycle2)
- GC content → Tempo and brightness
- Entropy → Note density and dynamics
- Tandem repeats → Motif looping
"""

import struct
import wave
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

# Try to import midiutil for MIDI generation
try:
    from midiutil import MIDIFile
    HAS_MIDIUTIL = True
except ImportError:
    HAS_MIDIUTIL = False


# =============================================================================
# SARGAM FREQUENCY RATIOS
# =============================================================================

# Just intonation ratios for Sargam (relative to Sa)
SARGAM_RATIOS = {
    'Sa':  1.0,       # Tonic
    'Re':  9/8,       # 1.125
    'Ga':  5/4,       # 1.25
    'Ma':  4/3,       # 1.333...
    'Pa':  3/2,       # 1.5
    'Dha': 5/3,       # 1.667...
    'Ni':  15/8,      # 1.875
    'Sa_high': 2.0,   # Octave
    'Re_high': 2.25,  # 9/4
    'Ga_high': 2.5,   # 5/2
    'Ma_high': 2.6666666666666665,   # 8/3
}

# New Chemical Mapping (G=Sa, A=Re, T=Ga, C=Ma)
# Odd/Even cycles mapped to Octaves for rhythmic variation
NUCLEOTIDE_SARGAM = {
    'G': {'odd': 'Sa', 'even': 'Sa_high'},  # Purine (Large) -> Root
    'A': {'odd': 'Re', 'even': 'Re_high'},  # Purine (Large) -> 2nd
    'T': {'odd': 'Ga', 'even': 'Ga_high'},  # Pyrimidine (Small) -> 3rd
    'C': {'odd': 'Ma', 'even': 'Ma_high'},  # Pyrimidine (Small) -> 4th
}

# Western Equal Temperament Ratios (12TET)
SEMITONE = 2 ** (1/12)
WESTERN_RATIOS = {
    'major': {
        'I':   1.0,
        'II':  SEMITONE ** 2,
        'III': SEMITONE ** 4,
        'IV':  SEMITONE ** 5,
        'V':   SEMITONE ** 7,
        'VI':  SEMITONE ** 9,
        'VII': SEMITONE ** 11,
        'Octave': 2.0
    },
    'minor': {
        'I':   1.0,
        'II':  SEMITONE ** 2,
        'bIII': SEMITONE ** 3,
        'IV':  SEMITONE ** 5,
        'V':   SEMITONE ** 7,
        'bVI': SEMITONE ** 8,
        'bVII': SEMITONE ** 10,
        'Octave': 2.0
    }
}

# Western Mapping
NUCLEOTIDE_WESTERN = {
    'major': {
        'A': {'odd': 'I', 'even': 'V'},
        'T': {'odd': 'II', 'even': 'VI'},
        'G': {'odd': 'III', 'even': 'VII'},
        'C': {'odd': 'IV', 'even': 'Octave'},
    },
    'minor': {
        'A': {'odd': 'I', 'even': 'V'},
        'T': {'odd': 'II', 'even': 'bVI'},
        'G': {'odd': 'bIII', 'even': 'bVII'},
        'C': {'odd': 'IV', 'even': 'Octave'},
    }
}

# Timbre profiles with harmonics, ADSR, and special effects
# Each instrument has a unique character beyond just harmonics
TIMBRE_PROFILES = {
    'sitar': {
        # Sitar: bright attack, buzzy sympathetic strings, long sustain
        'harmonics': [1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
        'attack': 0.02, 'decay': 0.1, 'sustain': 0.6, 'release': 0.5,
        'buzz': True,  # Add sympathetic string buzz
        'vibrato_rate': 5.0, 'vibrato_depth': 0.015
    },
    'harmonium': {
        # Harmonium: warm, reedy, organ-like with slight beating
        'harmonics': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2],
        'attack': 0.15, 'decay': 0.1, 'sustain': 0.85, 'release': 0.25,
        'detune': 0.003,  # Slight detuning for warmth
        'vibrato_rate': 0, 'vibrato_depth': 0
    },
    'veena': {
        # Veena: smooth, rich, singing quality
        'harmonics': [1.0, 0.6, 0.45, 0.35, 0.25, 0.15, 0.1],
        'attack': 0.05, 'decay': 0.15, 'sustain': 0.7, 'release': 0.4,
        'vibrato_rate': 4.5, 'vibrato_depth': 0.02
    },
    'bells': {
        # Bells/Glockenspiel: bright, inharmonic partials, fast decay
        'harmonics': [1.0, 0.0, 0.6, 0.0, 0.4, 0.0, 0.3],  # Odd harmonics
        'inharmonic': [1.0, 2.4, 3.0, 4.5, 5.2],  # Non-integer ratios
        'attack': 0.005, 'decay': 0.3, 'sustain': 0.2, 'release': 0.8,
        'vibrato_rate': 0, 'vibrato_depth': 0
    },
    'strings': {
        # String ensemble: lush, slow attack, rich harmonics
        'harmonics': [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
        'attack': 0.3, 'decay': 0.2, 'sustain': 0.75, 'release': 0.5,
        'detune': 0.004,  # Ensemble detuning
        'vibrato_rate': 5.5, 'vibrato_depth': 0.012
    },
    'organ': {
        # Pipe organ: full, sustained, no decay
        'harmonics': [1.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],  # Drawbar-like
        'attack': 0.08, 'decay': 0.05, 'sustain': 0.95, 'release': 0.15,
        'vibrato_rate': 6.0, 'vibrato_depth': 0.008  # Leslie-like
    },
    'lofi': {
        # Lofi Keys: Warm, wobbly, muted
        'harmonics': [1.0, 0.1, 0.05, 0.02],  # Mostly fundamental
        'attack': 0.05, 'decay': 0.3, 'sustain': 0.4, 'release': 0.4,
        'detune': 0.005, 'vibrato_rate': 2.0, 'vibrato_depth': 0.005, # Tape wobble
        'filter': 0.5 # Simulate low-pass (simple amp reduction on high harmonics)
    },
    'synth': {
        # Sawtooth-ish Lead
        'harmonics': [1.0, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.12], # 1/n series
        'attack': 0.005, 'decay': 0.1, 'sustain': 0.6, 'release': 0.1,
        'detune': 0.002
    }
}

# Legacy compatibility mapping
TIMBRE_HARMONICS = {k: v['harmonics'] for k, v in TIMBRE_PROFILES.items()}

# Map old names to new for backwards compatibility
TIMBRE_ALIASES = {
    'tanpura': 'harmonium',
    'flute': 'veena', 
    'pad': 'strings',
    'minimal': 'bells'
}


# =============================================================================
# SYNTHESIS ENGINE
# =============================================================================

class GenomeSynthesizer:
    """Synthesize DNA sequences into audio using Sargam, Western, or THz Physics mapping."""
    
    def __init__(self, 
                 sa_frequency: float = 261.63,  # C4 by default
                 sample_rate: int = 44100,
                 timbre: str = 'tanpura',
                 drone_enabled: bool = True,
                 scale_type: str = 'sargam',
                 # THz Physics Mode parameters
                 thz_k: int = 5,
                 thz_alpha: float = 0.35,
                 thz_mode: str = 'lowest',

                 thz_quantize: bool = False,
                 # Advanced Quantization
                 quant_type: str = 'nearest',
                 quant_strength: float = 0.7,
                 quant_smoothness: float = 0.6,
                 quant_policy: str = 'fundamental_only',

                 # Classical Overlays
                 overlay_pa: bool = False,
                 overlay_dha: bool = False,
                 overlay_ni: bool = False,
                 overlay_cpg: bool = False,
                 mix_pa: float = 0.25,
                 mix_dha: float = 0.35,
                 mix_ni: float = 0.25,
                 mix_cpg: float = 0.20):
        """
        Initialize synthesizer.
        
        Args:
            sa_frequency: Frequency of Sa/Root
            sample_rate: Audio sample rate
            timbre: Timbre profile
            drone_enabled: Enable background drone/pad
            scale_type: 'sargam', 'major', 'minor', or 'thz_physics'
            thz_k: Number of THz partials (3-8)
            thz_alpha: Compression exponent (0.2-0.6)
            thz_mode: Peak selection ('lowest', 'even', 'all')
            thz_quantize: Snap to Sargam scale
        """
        self.sa_freq = sa_frequency
        self.sample_rate = sample_rate
        self.timbre = timbre
        self.drone_enabled = drone_enabled
        self.scale_type = scale_type
        
        # THz Physics parameters
        self.thz_k = thz_k
        self.thz_alpha = thz_alpha
        self.thz_mode = thz_mode
        self.thz_quantize = thz_quantize
        
        # Advanced Quantization
        self.quant_type = quant_type
        self.quant_strength = quant_strength
        self.quant_smoothness = quant_smoothness
        self.quant_policy = quant_policy
        
        # Classical Overlays
        self.overlay_pa = overlay_pa
        self.overlay_dha = overlay_dha
        self.overlay_ni = overlay_ni
        self.overlay_cpg = overlay_cpg
        self.mix_pa = mix_pa
        self.mix_dha = mix_dha
        self.mix_ni = mix_ni
        self.mix_cpg = mix_cpg
        
        # Precompute note frequencies (not used in THz mode)
        self._compute_frequencies()
    
    def _compute_frequencies(self):
        """Compute all note frequencies based on Root/Sa."""
        self.frequencies = {}
        
        if self.scale_type == 'sargam':
            ratios = SARGAM_RATIOS
        elif self.scale_type in WESTERN_RATIOS:
            ratios = WESTERN_RATIOS[self.scale_type]
        else:
            ratios = SARGAM_RATIOS  # Default
            
        for note, ratio in ratios.items():
            self.frequencies[note] = self.sa_freq * ratio
    
    def _generate_adsr_envelope(self, duration: float, 
                                 attack: float = 0.05,
                                 decay: float = 0.1,
                                 sustain_level: float = 0.7,
                                 release: float = 0.1) -> np.ndarray:
        """
        Generate ADSR envelope.
        
        Args:
            duration: Total note duration in seconds
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain_level: Sustain amplitude (0-1)
            release: Release time in seconds
            
        Returns:
            Envelope array
        """
        samples = int(duration * self.sample_rate)
        envelope = np.zeros(samples)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            # Short note - just attack and release
            attack_samples = samples // 2
            release_samples = samples - attack_samples
            sustain_samples = 0
            decay_samples = 0
        
        idx = 0
        
        # Attack
        if attack_samples > 0:
            envelope[idx:idx + attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples
        
        # Decay
        if decay_samples > 0:
            envelope[idx:idx + decay_samples] = np.linspace(1, sustain_level, decay_samples)
            idx += decay_samples
        
        # Sustain
        if sustain_samples > 0:
            envelope[idx:idx + sustain_samples] = sustain_level
            idx += sustain_samples
        
        # Release
        if release_samples > 0:
            current_level = envelope[idx - 1] if idx > 0 else sustain_level
            envelope[idx:idx + release_samples] = np.linspace(current_level, 0, release_samples)
        
        return envelope
    
    def _synthesize_note(self, freq: float, duration: float, 
                          velocity: float = 0.8) -> np.ndarray:
        """
        Synthesize a single note with harmonics and envelope.
        
        Args:
            freq: Note frequency in Hz
            duration: Duration in seconds
            velocity: Loudness (0-1)
            
        Returns:
            Audio samples
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        
        # Resolve timbre alias if needed
        timbre_name = TIMBRE_ALIASES.get(self.timbre, self.timbre)
        profile = TIMBRE_PROFILES.get(timbre_name, TIMBRE_PROFILES['harmonium'])
        harmonics = profile['harmonics']
        
        # Get effect parameters
        vibrato_rate = profile.get('vibrato_rate', 0)
        vibrato_depth = profile.get('vibrato_depth', 0)
        detune = profile.get('detune', 0)
        has_buzz = profile.get('buzz', False)
        inharmonic = profile.get('inharmonic', None)
        
        # Apply vibrato to frequency if enabled
        if vibrato_rate > 0 and vibrato_depth > 0:
            vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        else:
            vibrato = 1.0
        
        # Generate waveform with harmonics
        wave = np.zeros(samples, dtype=np.float32)
        
        if inharmonic is not None:
            # Use inharmonic partials (for bells, etc.)
            for i, ratio in enumerate(inharmonic):
                partial_freq = freq * ratio * vibrato
                amp = harmonics[i] if i < len(harmonics) else 0.1
                # Inharmonic partials decay faster
                decay = np.exp(-t * (i + 1) * 2)
                wave += amp * np.sin(2 * np.pi * partial_freq * t) * decay
        else:
            # Standard harmonic synthesis
            for i, amplitude in enumerate(harmonics):
                harmonic_freq = freq * (i + 1) * vibrato
                
                # Apply detuning for ensemble/warmth effect
                if detune > 0:
                    # Create chorus effect with slightly detuned copies
                    wave += amplitude * 0.5 * np.sin(2 * np.pi * harmonic_freq * t)
                    wave += amplitude * 0.25 * np.sin(2 * np.pi * harmonic_freq * (1 + detune) * t)
                    wave += amplitude * 0.25 * np.sin(2 * np.pi * harmonic_freq * (1 - detune) * t)
                else:
                    wave += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add buzz for sitar-like instruments
        if has_buzz:
            # Sympathetic string resonance simulation
            buzz_freq = freq * 0.5  # Sub-octave buzz
            buzz = 0.15 * np.sin(2 * np.pi * buzz_freq * t)
            # Add high-frequency "jawari" buzz
            jawari = 0.08 * np.sin(2 * np.pi * freq * 7.5 * t) * np.exp(-t * 8)
            wave += buzz + jawari
        
        # Normalize
        wave = wave / np.max(np.abs(wave) + 1e-10)
        
        # Apply ADSR envelope with timbre-specific parameters
        envelope = self._generate_adsr_envelope(
            duration,
            attack=profile['attack'],
            decay=profile['decay'],
            sustain_level=profile['sustain'],
            release=profile['release']
        )
        wave = wave * envelope
        
        # Apply velocity
        wave = wave * velocity
        
        return wave

    def _synthesize_partials(self, partials: List[float], duration: float,
                              velocity: float = 0.8) -> np.ndarray:
        """
        Synthesize multiple simultaneous partials (THz Physics Mode).
        
        Creates a warm, chord-like timbre by summing sinusoids at
        different frequencies derived from THz spectroscopy data.
        
        Args:
            partials: List of frequencies in Hz
            duration: Duration in seconds
            velocity: Loudness (0-1)
        
        Returns:
            Audio samples
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        wave = np.zeros(samples, dtype=np.float32)
        
        if not partials:
            return wave
        
        # Sum partials with decreasing amplitude (1/i weighting)
        for i, freq in enumerate(partials):
            weight = 1.0 / (i + 1)
            # Add slight vibrato for warmth
            vibrato = 1 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
            wave += weight * np.sin(2 * np.pi * freq * vibrato * t)
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val
        
        # Apply warm ADSR envelope
        envelope = self._generate_adsr_envelope(
            duration,
            attack=0.08,
            decay=0.15,
            sustain_level=0.7,
            release=0.25
        )
        
        return wave * envelope * velocity
    
    def _generate_backing(self, duration: float, volume: float = 0.15) -> np.ndarray:
        """
        Generate background drone or chord pad.
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        backing = np.zeros(samples, dtype=np.float32)
        
        root_freq = self.sa_freq
        
        if self.scale_type == 'sargam':
            # Indian Drone: Sa + Pa (Root + 5th)
            pa_freq = root_freq * SARGAM_RATIOS['Pa']
            
            # Sa drone
            backing += np.sin(2 * np.pi * root_freq * t)
            backing += 0.5 * np.sin(2 * np.pi * root_freq * 2 * t)
            
            # Pa drone
            backing += 0.7 * np.sin(2 * np.pi * pa_freq * t)
            backing += 0.35 * np.sin(2 * np.pi * pa_freq * 2 * t)
            
        elif self.scale_type == 'major':
            # Major Pad: Root + 3rd + 5th (Major Triad)
            third_freq = root_freq * WESTERN_RATIOS['major']['III']
            fifth_freq = root_freq * WESTERN_RATIOS['major']['V']
            
            backing += np.sin(2 * np.pi * root_freq * t)
            backing += 0.8 * np.sin(2 * np.pi * third_freq * t)
            backing += 0.6 * np.sin(2 * np.pi * fifth_freq * t)
            
            # Add slow movement
            lfo = 1 + 0.05 * np.sin(2 * np.pi * 0.2 * t)
            backing *= lfo
            
        elif self.scale_type == 'minor':
            # Minor Pad: Root + b3rd + 5th (Minor Triad)
            third_freq = root_freq * WESTERN_RATIOS['minor']['bIII']
            fifth_freq = root_freq * WESTERN_RATIOS['minor']['V']
            
            backing += np.sin(2 * np.pi * root_freq * t)
            backing += 0.8 * np.sin(2 * np.pi * third_freq * t)
            backing += 0.6 * np.sin(2 * np.pi * fifth_freq * t)
        
        # Normalize and apply volume
        backing = backing / np.max(np.abs(backing) + 1e-10) * volume
        
        # Add Beat for Western modes
        if self.scale_type in ['major', 'minor']:
            beat_loop = self._generate_beat(duration)
            if len(beat_loop) > len(backing):
                beat_loop = beat_loop[:len(backing)]
            elif len(beat_loop) < len(backing):
                beat_loop = np.pad(beat_loop, (0, len(backing) - len(beat_loop)))
            backing += beat_loop
            
            # Normalize again
            backing = backing / np.max(np.abs(backing) + 1e-10) * volume * 1.5
        
        # Soft fade in/out
        fade_samples = int(0.5 * self.sample_rate)
        if samples > fade_samples * 2:
            backing[:fade_samples] *= np.linspace(0, 1, fade_samples)
            backing[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return backing

    def _make_kick(self) -> np.ndarray:
        """Generate 808-style kick."""
        duration = 0.3
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        # Pitch envelope (150Hz -> 50Hz)
        freq_env = 100 * np.exp(-t * 15) + 50
        # Amp envelope
        amp_env = np.exp(-t * 8)
        kick = np.sin(2 * np.pi * freq_env * t) * amp_env
        return kick * 0.8

    def _make_snare(self) -> np.ndarray:
        """Generate lofi snare (noise punch)."""
        duration = 0.15
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        # Noise
        noise = np.random.uniform(-1, 1, samples)
        # Envelope
        env = np.exp(-t * 20)
        # Low pass filter approximation (moving average)
        snare = np.convolve(noise * env, np.ones(5)/5, mode='same')
        return snare * 0.6

    def _make_hat(self) -> np.ndarray:
        """Generate hi-hat (high freq noise)."""
        duration = 0.05
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        noise = np.random.uniform(-1, 1, samples)
        # High pass approximation (diff)
        hat = np.diff(noise, prepend=0)
        env = np.exp(-t * 80)
        return hat * env * 0.4

    def _generate_beat(self, total_duration: float) -> np.ndarray:
        """Generate a drum loop repeated for duration."""
        bpm = 90  # Lofi Hip Hop tempo
        seconds_per_beat = 60 / bpm
        bar_duration = seconds_per_beat * 4
        
        bar_samples = int(bar_duration * self.sample_rate)
        bar = np.zeros(bar_samples, dtype=np.float32)
        
        kick = self._make_kick()
        snare = self._make_snare()
        hat = self._make_hat()
        
        # Pattern: Kick on 1, Snare on 3, Hats on 8ths
        # Beat 1
        pos = 0
        end = min(pos + len(kick), bar_samples)
        bar[pos:end] += kick[:end-pos]
        
        # Beat 3 (Snare)
        pos = int(seconds_per_beat * 2 * self.sample_rate)
        end = min(pos + len(snare), bar_samples)
        bar[pos:end] += snare[:end-pos]
        
        # Off-beat Kick (Syncopation) - Beat 2.5
        pos = int(seconds_per_beat * 1.5 * self.sample_rate)
        end = min(pos + len(kick), bar_samples)
        bar[pos:end] += kick[:end-pos] * 0.6
        
        # Hi-hats (every 0.5 beats)
        for i in range(8):
            pos = int(seconds_per_beat * 0.5 * i * self.sample_rate)
            end = min(pos + len(hat), bar_samples)
            # Add swing to off-beats
            if i % 2 == 1:
                pos += int(0.05 * self.sample_rate) # Swing
            if pos < bar_samples:
                end = min(pos + len(hat), bar_samples)
                bar[pos:end] += hat[:end-pos]
                
        # Tile to fill total duration
        total_samples = int(total_duration * self.sample_rate)
        full_beat = np.tile(bar, int(np.ceil(total_duration / bar_duration)))
        return full_beat[:total_samples]
    
    def sequence_to_notes(self, seq: str, 
                           base_duration: float = 0.35,  # Slower: was 0.15
                           gc_content: float = 50.0,
                           entropy: float = 2.0) -> List[Dict]:
        """
        Convert DNA sequence to note events.
        
        Args:
            seq: DNA sequence
            base_duration: Base note duration in seconds
            gc_content: GC% for tempo modulation
            entropy: Shannon entropy for density modulation
            
        Returns:
            List of note dicts with 'freq', 'duration', 'velocity', 'time'
        """
        notes = []
        counters = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        current_time = 0.0
        
        # Modulate tempo based on GC content (higher GC = slightly faster)
        # Range: 0.7 to 1.1 (was 0.5 to 1.3 - too aggressive)
        tempo_factor = 0.7 + (gc_content / 100) * 0.4
        note_duration = base_duration / tempo_factor
        
        # Modulate density based on entropy (higher entropy = denser notes)
        density_factor = 0.5 + (entropy / 2.0) * 0.5  # 0.5 to 1.0
        
        # Precompute overlays for Classical Mode
        mask_pa = None
        start_dha = set()
        mask_ni = None
        start_cpg = set()
        
        if self.scale_type == 'sargam':
            from . import classical_overlays
            if self.overlay_pa:
                mask_pa = classical_overlays.detect_coding_regions(seq)
            if self.overlay_dha:
                start_dha = set(classical_overlays.detect_repeats(seq))
            if self.overlay_ni:
                mask_ni = classical_overlays.detect_high_entropy(seq)
            if self.overlay_cpg:
                start_cpg = set(classical_overlays.detect_cpg_sites(seq))

        for i, nuc in enumerate(seq.upper()):
            if nuc not in 'ATGC':
                # Skip N and other characters
                current_time += note_duration * 0.5
                continue
            
            # Update counter
            counters[nuc] += 1
            
            # Determine cycle (odd or even)
            cycle = 'odd' if counters[nuc] % 2 == 1 else 'even'
            
            # Velocity based on entropy (lower entropy = softer)
            velocity = 0.4 + density_factor * 0.5
            
            # Get Note
            if self.scale_type == 'thz_physics':

                # THz Physics Mode: generate partials from spectroscopy data
                from . import thz_data, quantize
                
                # Get RAW partials (no quantization in thz_data)
                raw_partials = thz_data.get_base_partials(
                    nuc,
                    k=self.thz_k,
                    mode=self.thz_mode,
                    sa_hz=self.sa_freq,
                    alpha=self.thz_alpha,
                    quantize=False
                )
                
                partials = []
                if raw_partials:
                    if self.thz_quantize:
                        # Instantiate quantizer for this sequence
                        if 'quantizer' not in locals():
                            quantizer = quantize.SargamQuantizer(sa_hz=self.sa_freq)
                        
                        if self.quant_policy == 'fundamental_only':
                            # Policy 1: Quantize fundamental, preserve ratios
                            p0 = raw_partials[0]
                            p0_quant = quantizer.quantize(
                                p0, 
                                mode=self.quant_type, 
                                strength=self.quant_strength,
                                smoothness=self.quant_smoothness
                            )
                            # Apply ratio to other partials
                            ratio_base = p0_quant / p0 if p0 > 0 else 1.0
                            partials = [p * ratio_base for p in raw_partials]
                            
                        else:
                            # Policy 2: Quantize all partials independently
                            partials = [
                                quantizer.quantize(
                                    p,
                                    mode=self.quant_type,
                                    strength=self.quant_strength,
                                    smoothness=self.quant_smoothness
                                ) for p in raw_partials
                            ]
                    else:
                        partials = raw_partials
                
                notes.append({
                    'nucleotide': nuc,
                    'partials': partials,
                    'freq': partials[0] if partials else self.sa_freq,
                    'sargam': f'THz-{nuc}',
                    'duration': note_duration,
                    'velocity': velocity,
                    'time': current_time
                })
            elif self.scale_type == 'sargam':
                note_name = NUCLEOTIDE_SARGAM[nuc][cycle]
                freq = self.frequencies[note_name]
                notes.append({
                    'nucleotide': nuc,
                    'sargam': note_name,
                    'freq': freq,
                    'duration': note_duration,
                    'velocity': velocity,
                    'time': current_time
                })
                
                # Classical Overlays (Polyphony)
                # Pa (Coding Region): Continuous harmony
                if mask_pa is not None and i < len(mask_pa) and mask_pa[i]:
                    notes.append({
                        'nucleotide': 'Pa-Overlay',
                        'sargam': 'Pa',
                        'freq': self.sa_freq * 1.5,
                        'duration': note_duration,
                        'velocity': velocity * self.mix_pa,
                        'time': current_time
                    })
                
                # Dha (Repeats): Accent at start
                if self.overlay_dha and i in start_dha:
                    notes.append({
                        'nucleotide': 'Dha-Overlay',
                        'sargam': 'Dha',
                        'freq': self.sa_freq * 5/3,
                        'duration': note_duration * 0.5,
                        'velocity': velocity * self.mix_dha * 1.2, # Accent
                        'time': current_time
                    })

                # Ni (High Entropy): Tension harmony
                if mask_ni is not None and i < len(mask_ni) and mask_ni[i]:
                    notes.append({
                        'nucleotide': 'Ni-Overlay',
                        'sargam': 'Ni',
                        'freq': self.sa_freq * 15/8,
                        'duration': note_duration,
                        'velocity': velocity * self.mix_ni,
                        'time': current_time
                    })

                # CpG (Ornament): High shimmer
                if self.overlay_cpg and i in start_cpg:
                    notes.append({
                        'nucleotide': 'CpG-Overlay',
                        'sargam': 'Ni_high',
                        'freq': self.sa_freq * 15/8 * 2,
                        'duration': 0.1,
                        'velocity': self.mix_cpg,
                        'time': current_time
                    })
            elif self.scale_type in NUCLEOTIDE_WESTERN:
                note_name = NUCLEOTIDE_WESTERN[self.scale_type][nuc][cycle]
                freq = self.frequencies[note_name]
                notes.append({
                    'nucleotide': nuc,
                    'sargam': note_name,
                    'freq': freq,
                    'duration': note_duration,
                    'velocity': velocity,
                    'time': current_time
                })
            else:
                note_name = NUCLEOTIDE_SARGAM[nuc][cycle]
                freq = self.frequencies[note_name]
                notes.append({
                    'nucleotide': nuc,
                    'sargam': note_name,
                    'freq': freq,
                    'duration': note_duration,
                    'velocity': velocity,
                    'time': current_time
                })
            
            current_time += note_duration
        
        return notes
    
    def synthesize(self, seq: str, 
                   gc_content: float = 50.0,
                   entropy: float = 2.0) -> np.ndarray:
        """
        Synthesize DNA sequence to audio.
        """
        # Convert sequence to notes
        notes = self.sequence_to_notes(seq, gc_content=gc_content, entropy=entropy)
        
        if not notes:
            return np.zeros(self.sample_rate, dtype=np.float32)
        
        # Calculate total duration
        last_note = notes[-1]
        total_duration = last_note['time'] + last_note['duration'] + 0.5
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize audio buffer
        audio = np.zeros(total_samples, dtype=np.float32)
        
        # Synthesize each note
        for note in notes:
            start_sample = int(note['time'] * self.sample_rate)
            
            # Check if note has partials (THz mode) or single freq
            if 'partials' in note and note['partials']:
                note_audio = self._synthesize_partials(
                    note['partials'],
                    note['duration'],
                    note['velocity']
                )
            else:
                note_audio = self._synthesize_note(
                    note['freq'], 
                    note['duration'], 
                    note['velocity']
                )
            
            # Add to buffer
            end_sample = min(start_sample + len(note_audio), total_samples)
            audio[start_sample:end_sample] += note_audio[:end_sample - start_sample]
        
        # Add backing if enabled
        if self.drone_enabled:
            backing = self._generate_backing(total_duration)
            if len(backing) > len(audio):
                backing = backing[:len(audio)]
            elif len(backing) < len(audio):
                backing = np.pad(backing, (0, len(audio) - len(backing)))
            audio = audio + backing
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def save_wav(self, audio: np.ndarray, filepath: str):
        """
        Save audio to WAV file.
        
        Args:
            audio: Audio samples (float32)
            filepath: Output path
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())


# =============================================================================
# MIDI GENERATION
# =============================================================================

def sargam_to_midi_note(sargam: str, base_midi: int = 60) -> int:
    """
    Convert Sargam note to MIDI note number.
    
    Args:
        sargam: Sargam note name
        base_midi: MIDI note for Sa (default 60 = C4)
        
    Returns:
        MIDI note number
    """
    # Semitone offsets from Root/Sa
    offsets = {
        # Sargam
        'Sa': 0, 'Re': 2, 'Ga': 4, 'Ma': 5,
        'Pa': 7, 'Dha': 9, 'Ni': 11, 'Sa_high': 12,
        # Western Major
        'I': 0, 'II': 2, 'III': 4, 'IV': 5,
        'V': 7, 'VI': 9, 'VII': 11, 'Octave': 12,
        # Western Minor (Natural)
        'bIII': 3, 'bVI': 8, 'bVII': 10
    }
    return base_midi + offsets.get(sargam, 0)


def sequence_to_midi(seq: str, 
                     output_path: str,
                     tempo: int = 120,
                     base_midi: int = 60) -> bool:
    """
    Convert DNA sequence to MIDI file.
    
    Args:
        seq: DNA sequence
        output_path: Output MIDI file path
        tempo: BPM
        base_midi: MIDI note for Sa
        
    Returns:
        True if successful
    """
    if not HAS_MIDIUTIL:
        return False
    
    midi = MIDIFile(1)  # One track
    track = 0
    channel = 0
    time = 0
    duration = 0.5  # Quarter note
    volume = 100
    
    midi.addTempo(track, 0, tempo)
    
    counters = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    
    for nuc in seq.upper():
        if nuc not in 'ATGC':
            time += 0.25
            continue
        
        counters[nuc] += 1
        cycle = 'odd' if counters[nuc] % 2 == 1 else 'even'
        sargam = NUCLEOTIDE_SARGAM[nuc][cycle]
        
        midi_note = sargam_to_midi_note(sargam, base_midi)
        midi.addNote(track, channel, midi_note, time, duration, volume)
        
        time += duration
    
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    return True


# =============================================================================
# PIANO ROLL VISUALIZATION DATA
# =============================================================================

def get_piano_roll_data(seq: str, 
                        gc_content: float = 50.0,
                        entropy: float = 2.0) -> List[Dict]:
    """
    Get piano roll visualization data.
    
    Args:
        seq: DNA sequence
        gc_content: GC% for tempo
        entropy: Shannon entropy
        
    Returns:
        List of note events for visualization
    """
    synth = GenomeSynthesizer()
    notes = synth.sequence_to_notes(seq, gc_content=gc_content, entropy=entropy)
    
    # Add MIDI-like pitch for visualization
    for note in notes:
        # Check if 'sargam' key exists (it might be named differently in future)
        note_name = note.get('sargam', 'Sa')
        note['pitch'] = sargam_to_midi_note(note_name)
    
    return notes


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def generate_music(seq: str,
                   sa_frequency: float = 261.63,
                   timbre: str = 'tanpura',
                   drone_enabled: bool = True,
                   gc_content: Optional[float] = None,
                   entropy: Optional[float] = None,
                   scale_type: str = 'sargam',
                   # THz Physics Mode parameters
                   thz_k: int = 5,
                   thz_alpha: float = 0.35,
                   thz_mode: str = 'lowest',
                   thz_quantize: bool = False,
                   # Advanced Quantization
                   quant_type: str = 'nearest',
                   quant_strength: float = 0.7,
                   quant_smoothness: float = 0.6,
                   quant_policy: str = 'fundamental_only',
                   # Classical Overlays
                   overlay_pa: bool = False,
                   overlay_dha: bool = False,
                   overlay_ni: bool = False,
                   overlay_cpg: bool = False,
                   mix_pa: float = 0.25,
                   mix_dha: float = 0.35,
                   mix_ni: float = 0.25,
                   mix_cpg: float = 0.20) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate music from DNA sequence.
    
    Args:
        seq: DNA sequence
        sa_frequency: Sa frequency in Hz
        timbre: Timbre profile
        drone_enabled: Enable drone
        gc_content: Override GC% (auto-calculated if None)
        entropy: Override entropy (auto-calculated if None)
        scale_type: 'sargam', 'major', 'minor', or 'thz_physics'
        thz_k: Number of THz partials (3-8) [Physics Mode]
        thz_alpha: Compression exponent (0.2-0.6) [Physics Mode]
        thz_mode: Peak selection ('lowest', 'even') [Physics Mode]
        thz_quantize: Snap to Sargam scale [Physics Mode]
        quant_type: 'nearest', 'soft', 'smart' [Adv Quant]
        quant_strength: 0-1 for soft mode
        quant_smoothness: 0-1 for smart mode
        quant_policy: 'fundamental_only' or 'all'
        overlay_pa: Enable Pa (Perfect Fifth) overlay
        overlay_dha: Enable Dha (Major Sixth) overlay
        overlay_ni: Enable Ni (Major Seventh) overlay
        overlay_cpg: Enable CpG island overlay
        mix_pa: Mixing level for Pa overlay (0-1)
        mix_dha: Mixing level for Dha overlay (0-1)
        mix_ni: Mixing level for Ni overlay (0-1)
        mix_cpg: Mixing level for CpG overlay (0-1)
        
    Returns:
        (audio_samples, note_events)
    """
    from . import stats
    
    # Calculate stats if not provided
    if gc_content is None:
        gc_content = stats.calculate_gc_content(seq)
    if entropy is None:
        entropy = stats.calculate_entropy(seq)
    
    # Create synthesizer
    synth = GenomeSynthesizer(
        sa_frequency=sa_frequency,
        timbre=timbre,
        drone_enabled=drone_enabled,
        scale_type=scale_type,
        thz_k=thz_k,
        thz_alpha=thz_alpha,
        thz_mode=thz_mode,
        thz_quantize=thz_quantize,
        quant_type=quant_type,
        quant_strength=quant_strength,
        quant_smoothness=quant_smoothness,
        quant_policy=quant_policy,
        overlay_pa=overlay_pa,
        overlay_dha=overlay_dha,
        overlay_ni=overlay_ni,
        overlay_cpg=overlay_cpg,
        mix_pa=mix_pa,
        mix_dha=mix_dha,
        mix_ni=mix_ni,
        mix_cpg=mix_cpg
    )
    
    # Synthesize
    audio = synth.synthesize(seq, gc_content=gc_content, entropy=entropy)
    notes = synth.sequence_to_notes(seq, gc_content=gc_content, entropy=entropy)
    
    return audio, notes


def save_music(audio: np.ndarray, 
               output_path: str,
               sample_rate: int = 44100):
    """Save audio to WAV file."""
    synth = GenomeSynthesizer(sample_rate=sample_rate)
    synth.save_wav(audio, output_path)


def get_mapping_legend() -> str:
    """Get human-readable mapping legend."""
    return """
## DNA → Music Mapping (Sargam System)

### Nucleotide → Note (Occurrence Toggle)
| Nucleotide | Odd Occurrence | Even Occurrence |
|------------|----------------|-----------------|
| **G** (Guanine) | Sa (Root) | Sa' (Octave) |
| **A** (Adenine) | Re (Second) | Re' (High 2nd) |
| **T** (Thymine) | Ga (Third) | Ga' (High 3rd) |
| **C** (Cytosine) | Ma (Fourth) | Ma' (High 4th) |

### Biological Modulation
| Feature | Musical Effect |
|---------|----------------|
| GC Content ↑ | Faster tempo, brighter tone |
| Entropy ↑ | Denser notes, higher velocity |
| Tandem Repeats | Looping motifs |

### Timbre Profiles
- **Tanpura**: Warm, meditative drone (Indian classical)
- **Flute**: Soft, breathy texture
- **Pad**: Cinematic, lush harmonics
- **Minimal**: Clean, analytical tones
"""
