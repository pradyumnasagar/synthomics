"""
Audio Visualization utilities using numpy and matplotlib.
"""

import io
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from typing import Tuple, Union


def read_wav(data: Union[str, bytes, io.BytesIO]) -> Tuple[int, np.ndarray]:
    """Read WAV data into (sample_rate, float32_array)."""
    if isinstance(data, (bytes, io.BytesIO)):
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        elif hasattr(data, "seek"):
            data.seek(0)
        sr, samples = wav.read(data)
    else:
        sr, samples = wav.read(data)
        
    # Convert to mono if stereo
    if len(samples.shape) == 2:
        samples = samples.mean(axis=1)
        
    # Normalize to -1.0 to 1.0 float32
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
        
    return sr, samples


def plot_waveform(samples: np.ndarray, sr: int, title: str = "Waveform"):
    """Plot basic waveform with downsampling for performance."""
    max_points = 100000
    if len(samples) > max_points:
        ds_factor = len(samples) // max_points
        plot_samples = samples[::ds_factor]
        time_axis = np.arange(len(plot_samples)) * ds_factor / sr
    else:
        plot_samples = samples
        time_axis = np.arange(len(samples)) / sr

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, plot_samples, color='#1f77b4', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_waveform_comparison(samples1: np.ndarray, samples2: np.ndarray, sr: int, title1: str, title2: str, title: str = "Waveform Comparison"):
    """Plot two waveforms in vertically stacked subplots for easy visual comparison."""
    max_points = 100000
    
    def process(samples):
        if len(samples) > max_points:
            ds_factor = len(samples) // max_points
            p_samples = samples[::ds_factor]
            t_axis = np.arange(len(p_samples)) * ds_factor / sr
        else:
            p_samples = samples
            t_axis = np.arange(len(samples)) / sr
        return t_axis, p_samples
        
    t1, p1 = process(samples1)
    t2, p2 = process(samples2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    
    # Original Sequence
    ax1.plot(t1, p1, color='#1f77b4', linewidth=0.5)
    ax1.set_title(f"1: {title1}", fontsize=10)
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Modified/Comparison Sequence
    ax2.plot(t2, p2, color='#d62728', linewidth=0.5)
    ax2.set_title(f"2: {title2}", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    
    ax1.set_ylim(-1.0, 1.0)
    plt.tight_layout()
    return fig


def plot_spectrogram(samples: np.ndarray, sr: int, title: str = "Spectrogram"):
    """Plot spectrogram using matplotlib.specgram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    # specgram requires 1D array
    Pxx, freqs, bins, im = ax.specgram(samples, NFFT=1024, Fs=sr, noverlap=512, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, sr / 2)
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_rms_energy(samples: np.ndarray, sr: int, title: str = "RMS Energy", frame_length: int = 2048, hop_length: int = 512):
    """Plot Root Mean Square energy over time."""
    # Compute RMS using simple sliding window
    # Ensure divisible size or pad
    n_frames = 1 + (len(samples) - frame_length) // hop_length
    if n_frames < 1:
        n_frames = 1
        rms = np.array([np.sqrt(np.mean(samples**2))])
    else:
        # Strided array trick for rolling window could be used, but simple loop is very fast in numpy if vectorized properly
        # Or reshape based approach:
        shape = (n_frames, frame_length)
        strides = (samples.strides[0] * hop_length, samples.strides[0])
        frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames**2, axis=1))

    time_axis = np.arange(n_frames) * hop_length / sr

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, rms, color='#ff7f0e', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def get_wavesurfer_html(base64_audio: str) -> str:
    """Generate HTML block containing an interactive WaveSurfer.js player mapped to Base64 audio."""
    return f"""
    <html>
    <head>
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: sans-serif; }}
      #waveform {{ width: 100%; height: 80px; margin-bottom: 10px; cursor: pointer; }}
      .controls-container {{ display: flex; align-items: center; justify-content: center; gap: 15px; }}
      .btn {{ background-color: #ff4b4b; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
      .btn:hover {{ background-color: #ff2b2b; transform: scale(1.02); }}
      .speed-control {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #444; }}
      input[type=range] {{ width: 100px; }}
    </style>
    </head>
    <body>
    <div id="waveform"></div>
    <div class="controls-container">
      <button class="btn" id="playBtn">▶ Play/Pause</button>
      <div class="speed-control">
        <label for="speedSlider">Speed: <span id="speedVal">1.0x</span></label>
        <input type="range" id="speedSlider" min="0.25" max="2.0" step="0.25" value="1.0">
      </div>
    </div>
    <script type="module">
    import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
    const wavesurfer = WaveSurfer.create({{
      container: '#waveform',
      waveColor: '#ff9090',
      progressColor: '#ff4b4b',
      url: 'data:audio/wav;base64,{base64_audio}',
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    }});
    
    // Play/Pause binding
    const playBtn = document.getElementById('playBtn');
    playBtn.onclick = () => wavesurfer.playPause();
    
    // Playback Speed Slider binding
    const speedSlider = document.getElementById('speedSlider');
    const speedVal = document.getElementById('speedVal');
    speedSlider.oninput = (e) => {{
        const rate = parseFloat(e.target.value);
        wavesurfer.setPlaybackRate(rate);
        speedVal.innerText = rate.toFixed(2) + 'x';
    }};
    </script>
    </body>
    </html>
    """

