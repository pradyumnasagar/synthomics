# SoundFont Information

## Required SoundFont

The app generates audio using pure Python synthesis (no external SoundFont required).

However, if you want to use FluidSynth for MIDI playback, you'll need a SoundFont file.

## Recommended: FluidR3_GM

**File**: `FluidR3_GM.sf2` (~140 MB)
**License**: MIT (free for any use)

### Download Links:
- Ubuntu/Debian: `sudo apt install fluid-soundfont-gm`
  - Installs to: `/usr/share/sounds/sf2/FluidR3_GM.sf2`
  
- Manual download: https://member.keymusician.com/Member/FluidR3_GM/

## Note

The Genome Sonics app does NOT require FluidSynth or SoundFonts.
It generates WAV audio directly using harmonic synthesis.
This file is only needed if you want to use external MIDI players.
