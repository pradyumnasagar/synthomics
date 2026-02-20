"""
Difference sonification module for generating glitch and pulse layers.
"""

import numpy as np
import scipy.io.wavfile as wav
import io
import re

def generate_difference_track(
    seq1: str,
    seq2: str,
    base_audio_buffer: io.BytesIO,
    note_duration: float = 0.125, # 120BPM default
    do_glitch: bool = True,
    do_pulse: bool = True
) -> io.BytesIO:
    """
    Generate a difference track highlighting mutations and repeats.
    Reads a base_audio_buffer (e.g. from seq2) and overlays pings/clicks.
    """
    base_audio_buffer.seek(0)
    sr, base_audio = wav.read(base_audio_buffer)
    
    # Ensure float32 representation
    if base_audio.dtype == np.int16:
        base_audio = base_audio.astype(np.float32) / 32768.0
    elif base_audio.dtype == np.int32:
        base_audio = base_audio.astype(np.float32) / 2147483648.0
        
    if len(base_audio.shape) > 1:
         base_audio = base_audio.mean(axis=1) # force mono
    
    out_audio = base_audio * 0.3 # Reduce volume of base track
    
    total_samples = len(out_audio)
    
    # Generate timestamp mapping variables
    samples_per_note = int(note_duration * sr)
    
    if do_glitch:
        # Align sequences (truncate to shortest for simple index match)
        min_len = min(len(seq1), len(seq2))
        mismatches = [i for i in range(min_len) if seq1[i] != seq2[i]]
        
        # Super high frequency ping parameter
        glitch_freq = 6000.0
        glitch_dur_samples = int(0.05 * sr)
        t_glitch = np.arange(glitch_dur_samples) / sr
        glitch_wave = np.sin(2 * np.pi * glitch_freq * t_glitch) * np.exp(-t_glitch * 40) # Sharp decay
        
        for m in mismatches:
            start_idx = int(m * samples_per_note)
            end_idx = start_idx + glitch_dur_samples
            if end_idx <= total_samples:
                out_audio[start_idx:end_idx] += glitch_wave * 0.6
                
    if do_pulse:
        # Detect dominant repeats in Seq2
        motifs = ["CAG", "CTG", "CGG"]
        
        for motif in motifs:
            matches = list(re.finditer(f"((?:{motif})+)", seq2))
            for match in matches:
                # the match is the full block: e.g. "CAGCAGCAG"
                run_length = len(match.group(1)) // 3
                if run_length >= 5: # only pulse if it's an actual structural run
                    start_base = match.start()
                    
                    # Low frequency percussive click
                    pulse_freq = 150.0  # Thump
                    pulse_dur_samples = int(0.08 * sr)
                    t_pulse = np.arange(pulse_dur_samples) / sr
                    pulse_wave = np.sin(2 * np.pi * pulse_freq * t_pulse + (10 * np.exp(-t_pulse*100))) * np.exp(-t_pulse * 30)
                    
                    # Add a pulse for each repeat unit (every 3 bases)
                    for i in range(run_length):
                        event_base = start_base + (i * 3)
                        start_idx = int(event_base * samples_per_note)
                        end_idx = start_idx + pulse_dur_samples
                        if end_idx <= total_samples:
                            out_audio[start_idx:end_idx] += pulse_wave * 0.8

    # Prevent clipping
    max_val = np.max(np.abs(out_audio))
    if max_val > 1.0:
        out_audio /= max_val
        
    out_audio = np.clip(out_audio, -1.0, 1.0)
    
    # Convert back to int16 bytes
    out_int16 = (out_audio * 32767).astype(np.int16)
    out_buffer = io.BytesIO()
    wav.write(out_buffer, sr, out_int16)
    out_buffer.seek(0)
    
    return out_buffer
