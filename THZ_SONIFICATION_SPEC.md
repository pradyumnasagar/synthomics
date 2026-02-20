# THz Physics Mode: Sonification Specification

> **Version**: 1.0  
> **Date**: 2026-01-31

## Overview

Physics Mode uses published THz (terahertz) absorption spectroscopy data to create 
a scientifically-grounded DNA sonification. Instead of arbitrary nucleotide→note 
mapping, frequencies are derived from measured molecular fingerprints.

---

## Data Source

### Primary Citation
**Yu, L. et al. (2019)**  
"Terahertz Absorption Spectroscopy of DNA Nucleobases for Plant Genomes Identification"  
*Sensors*, 19(5), 1148  
DOI: [10.3390/s19051148](https://doi.org/10.3390/s19051148)

### Foundational Reference
**Fischer, B.M. et al. (2002)**  
"Far-infrared vibrational modes of DNA components studied by terahertz time-domain spectroscopy"  
*Physics in Medicine and Biology*, 47(21), 3807

---

## THz Peak Data (Yu et al. 2019, Table 2)

| Nucleobase | THz Peak Positions |
|------------|-------------------|
| **Adenine (A)** | 1.70, 2.15, 2.50, 3.10, 4.10, 5.60, 6.00, 7.30 THz |
| **Guanine (G)** | 2.50, 3.00, 4.30, 4.80, 5.35, 6.30, 7.20, 9.80 THz |
| **Cytosine (C)** | 1.55, 2.75, 3.40, 4.35, 4.75, 5.95, 6.95 THz |
| **Thymine (T)** | 1.30, 2.25, 2.95, 4.50, 5.10, 6.30, 8.50, 9.60 THz |

---

## Mapping Mathematics

### Ratio-Preserving Compression

THz frequencies (10¹² Hz) are compressed to audible range (~100-1000 Hz) while 
preserving relative ratios:

```
p_min = 1.30 THz  (global minimum, from Thymine)

For each THz peak p:
    ratio = p / p_min
    f_audible = Sa_Hz × (ratio ^ alpha)
```

### Default Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Sa_Hz` | 261.63 | 110-440 | Base frequency (C4) |
| `alpha` | 0.35 | 0.20-0.60 | Compression exponent |
| `K` | 5 | 3-8 | Number of partials per note |

### Example Calculation (Adenine, default params)

| THz Peak | Ratio (p/1.30) | Audible Hz |
|----------|----------------|------------|
| 1.70 | 1.31 | 289 |
| 2.15 | 1.65 | 308 |
| 2.50 | 1.92 | 326 |
| 3.10 | 2.38 | 350 |
| 4.10 | 3.15 | 385 |

---

## Optional Quantization

When "Enable Quantization" is selected, the system offers three modes:

### 1. Nearest (Classic)
Snaps frequencies to the absolute nearest Sargam scale degree. Can feel jumpy.

### 2. Soft (Recommended)
Blends the raw THz frequency with the snapped target to retain microtonal character while suggesting the scale. Consonance without rigidity.
`f_out = f_raw * (1-strength) + f_snapped * strength`

### 3. Smart (Continuity-Aware)
Uses a cost function to minimize large interval jumps, prioritizing melodic smoothness over absolute nearest-neighbor snapping.
`Cost = Dist(target, input) + w * Dist(target, prev_note)`

### Partial Policy
- **Fundamental Only (Default)**: Quantizes only the lowest partial (`f0`). Upper partials are shifted by the same ratio `f0_quant / f0`. This **preserves the exact THz spectral fingerprint** (timbre) while tuning the pitch.
- **Quantize All**: Snaps every partial independently. Results in a more "perfect" harmonic sound but loses the specific THz color.

---

## Sound Design

### Timbre Generation

Each nucleotide event produces a **chord-like** sound by summing K sinusoids 
at the mapped partial frequencies:

```python
wave = Σ (1/(i+1)) × sin(2π × partial_i × t)
```

### Envelope (ADSR)

| Parameter | Value | Effect |
|-----------|-------|--------|
| Attack | 0.08s | Smooth onset |
| Decay | 0.15s | Natural fall |
| Sustain | 0.70 | Maintained level |
| Release | 0.25s | Gentle fade |

### Warmth Features
- Subtle vibrato (0.3% depth, 4.5 Hz rate)
- Normalization to prevent clipping
- Compatible with existing timbre profiles

---

## UI Controls

| Control | Type | Default | Description |
|---------|------|---------|-------------|
| Music System | Radio | Sargam | Select "🔬 Physics Mode (THz)" |
| Partials (K) | Slider | 5 | THz peaks per nucleotide |
| Compression (α) | Slider | 0.35 | Frequency spread |
| Quantize | Checkbox | OFF | Snap to Sargam scale |

---

## Scientific Disclaimer

> **IMPORTANT**: Molecular vibrations occur at THz frequencies (10¹² Hz), 
> approximately 50 billion times higher than human hearing range (20-20,000 Hz).
>
> This mode maps *published THz absorption peak positions* into the audible 
> range via ratio-preserving compression for **DATA SONIFICATION**.
>
> **We are NOT "hearing molecules"**—we are using spectroscopic fingerprints 
> as frequency input to create a scientifically-inspired audio representation.

---

## Methods Paragraph (Poster-Ready)

**Physics Mode Sonification**: We implemented a THz spectroscopy-based DNA 
sonification using absorption peak positions from Yu et al. (Sensors 2019). 
Each nucleobase (A/T/G/C) has a distinct THz fingerprint comprising 7-8 peaks 
between 1.3-9.8 THz. These peaks are mapped to audible frequencies via ratio-
preserving compression: f_audible = Sa × (p/p_min)^α, where p_min = 1.30 THz 
(Thymine), Sa = 261.63 Hz, and α = 0.35. For each nucleotide event, K peaks 
are selected and rendered simultaneously as partials with 1/i amplitude 
weighting, creating a chord-like timbre. An optional quantization mode snaps 
frequencies to Sargam scale degrees. This approach grounds the sonification in 
measured physical data rather than arbitrary mapping, while clearly disclaiming 
that THz vibrations are not directly audible—the output is data sonification, 
not molecular sound.

---

## Files Modified

| File | Changes |
|------|---------|
| `genome_sonics/thz_data.py` | NEW: THz peak data, mapping functions, citations |
| `genome_sonics/music.py` | Added `_synthesize_partials()`, THz params in `GenomeSynthesizer` |
| `app.py` | Added Physics Mode UI controls and disclaimer |
