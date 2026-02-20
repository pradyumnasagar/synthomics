# 🧬 SynthOmics

> Transform DNA sequences into deterministic music (Indian Classical / Western) and generative visual art.
> A Science Day interactive demo tool.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🎵 Music Systems**:
  - **🕉️ Indian Classical (Sargam)**: Just Intonation ratios with Tanpura drone.
  - **🎼 Western Mode**: Equal Temperament (Major/Minor) with Chord Pads.
  - **🧢 Gen Z Mode**: Procedural **Lo-fi / Trap Beats** integration.
  - **🔬 Physics Mode**: THz spectroscopy-based mapping (Yu et al. 2019).

- **🎹 8 Distinct Instruments**:
  - **Traditional**: Sitar, Harmonium, Veena, Bells
  - **Orchestral**: Strings, Pipe Organ
  - **Modern**: Lofi Keys (w/ tape wobble), Synth Lead (EDM)

- **🧬 DNA Mapping**: 
  - Occurrence-toggle mapping (DNA → Notes)
  - GC content modulates tempo
  - Entropy modulates note density & velocity

- **🎨 Generative Art**: Three unique visual styles
  - Heatmap + DNA Walk (scientific)
  - Entropy Chaos (organic)
  - Codon Mosaic (geometric)

- **📊 Sequence Analysis**: GC%, entropy, k-mer frequencies, tandem repeats

- **🔬 Compare Mode**: Side-by-side comparison with similarity metrics

- **📥 Export**: WAV, MIDI, PNG, and ZIP package

## 🚀 Quick Start

### WSL2 / Ubuntu
```bash
cd genome_sonics
bash setup_wsl.sh
source venv/bin/activate
streamlit run app.py
```

### Windows
```powershell
cd genome_sonics
.\setup_windows.bat
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## 📁 Project Structure

```
genome_sonics/
├── app.py                  # Streamlit main app
├── genome_sonics/          # Core package
│   ├── io.py               # FASTA parsing
│   ├── stats.py            # Sequence statistics
│   ├── music.py            # Synthesis Engine (Sargam + Western)
│   ├── art.py              # Art generation
│   └── compare.py          # Similarity metrics
├── requirements.txt
├── setup_wsl.sh
├── setup_windows.bat
└── README.md
```

## 🎵 Music Mapping Systems

### 1. Indian Classical (Sargam)
Uses strict **Just Intonation** ratios.
| Nucleotide | Odd Occurrence | Even Occurrence |
|------------|----------------|-----------------|
| **G** (Purine) | Sa (Root) | Sa' (Octave) |
| **A** (Purine) | Re (2nd) | Re' (High 2nd) |
| **T** (Pyrimidine) | Ga (3rd) | Ga' (High 3rd) |
| **C** (Pyrimidine) | Ma (4th) | Ma' (High 4th) |

*Backing: Tanpura Drone (Sa + Pa)*

### 2. Western (Major / Minor)
Uses **12-Tone Equal Temperament** (Piano tuning).
| Nucleotide | Odd Occurrence | Even Occurrence |
|------------|----------------|-----------------|
| A | I (Root) | V (Fifth) |
| T | II (2nd) | VI (6th) |
| G | III (3rd) | VII (7th) |
| C | IV (4th) | Octave |

*Backing: Chord Pads (Triads) + Lo-fi/Trap Beats*

### Biological Modulation
| Feature | Effect |
|---------|--------|
| GC Content ↑ | Faster tempo |
| Entropy ↑ | Denser notes, higher velocity |

## 🎨 Art Styles

1. **Heatmap + Walk**: K-mer frequency heatmap with DNA walk trajectory
2. **Entropy Chaos**: Chaos game with entropy-driven coloring
3. **Codon Mosaic**: 64-color grid based on codon identity

## 🔬 Demo Sequences

Built-in sequences for instant demo:
- HPV-16 E6/E7 (virus)
- SARS-CoV-2 Spike (virus)
- E. coli 16S rRNA (bacteria)
- BRCA1 Snippet (human)
- Random DNA (synthetic)

## 🎤 Demo Script

### Opening (30 seconds)
> "DNA is information encoded in four letters. This tool translates the language of life into music and art."

### Demo Flow (3 minutes)
1. **Load HPV-16**: Play it in **Sargam mode** with **Sitar**.
   > "Hear the repetitive motifs of viral DNA in a classical Indian scale."
2. **Switch to Western Minor**: Change instrument to **Lofi Keys**.
   > "Same DNA, but now in a modern Lo-fi context. The structure remains, the vibe changes."
3. **Switch to Synth Lead**: Enable the **Gen Z Trap beat**.
   > "Data sonification doesn't have to be boring. It can be a banger."
4. **Compare Tab**: HPV vs SARS-CoV-2.
   > "Different organisms have different signatures."

### Closing (30 seconds)
> "Every genome has its own song. We're just learning to listen."

## 🔧 Troubleshooting

### FluidSynth not found
The app generates WAV audio directly without FluidSynth. MIDI export still works.

### Large FASTA slow
Sequences >50kb are automatically sampled to maintain performance.

### No audio in browser
Check browser audio permissions. Try a different browser.

## 📄 License

MIT License - Free for educational and research use.

---

<p align="center">
  <b>🧬 DNA is information. We experience it through sound and sight.</b>
</p>
