"""
SynthOmics
==========
A Science Day demo tool that converts DNA sequences into 
deterministic Sargam-based music and generative visual art.

Run with: streamlit run app.py
"""

import os
import io
import sys
import tempfile
import base64
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from genome_sonics import io as gs_io
from genome_sonics import stats as gs_stats
from genome_sonics import art as gs_art
from genome_sonics import compare as gs_compare
from genome_sonics import music as gs_music
from genome_sonics import audio_viz as gs_viz
from genome_sonics import diff_audio as gs_diff
from genome_sonics import music as gs_music
from genome_sonics import art as gs_art
from genome_sonics import compare as gs_compare

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SynthOmics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f2027 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95);
        border-right: 1px solid rgba(124, 58, 237, 0.3);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(30, 30, 60, 0.8);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 60, 0.5);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 60, 0.5);
        border-radius: 8px;
    }
    
    /* Piano roll container */
    .piano-roll {
        background: rgba(20, 20, 40, 0.9);
        border-radius: 8px;
        padding: 10px;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if 'sequence' not in st.session_state:
    st.session_state.sequence = None
if 'sequence_name' not in st.session_state:
    st.session_state.sequence_name = None
if 'audio' not in st.session_state:
    st.session_state.audio = None
if 'artwork' not in st.session_state:
    st.session_state.artwork = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'notes' not in st.session_state:
    st.session_state.notes = None

# Music settings state
if 'sa_frequency' not in st.session_state:
    st.session_state.sa_frequency = 261  # C4 default

# Compare mode state
if 'seq1' not in st.session_state:
    st.session_state.seq1 = None
if 'seq2' not in st.session_state:
    st.session_state.seq2 = None
if 'seq1_name' not in st.session_state:
    st.session_state.seq1_name = None
if 'seq2_name' not in st.session_state:
    st.session_state.seq2_name = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def audio_to_base64(audio: np.ndarray, sample_rate: int = 44100) -> str:
    """Convert audio array to base64 for HTML audio player."""
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Write to WAV in memory
    buffer = io.BytesIO()
    import wave
    with wave.open(buffer, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def image_to_base64(img) -> str:
    """Convert PIL image to base64."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def render_piano_roll(notes: list, max_notes: int = 200):
    """Render piano roll visualization."""
    if not notes:
        st.info("No notes to display")
        return
    
    # Limit notes for performance
    display_notes = notes[:max_notes]
    
    # Create SVG piano roll
    width = min(len(display_notes) * 8, 1200)
    height = 150
    
    # Note to y-position mapping (MIDI-like)
    min_pitch = min(n.get('pitch', 60) for n in display_notes)
    max_pitch = max(n.get('pitch', 72) for n in display_notes)
    pitch_range = max(max_pitch - min_pitch, 12)
    
    svg_notes = []
    for i, note in enumerate(display_notes):
        x = i * 8
        pitch = note.get('pitch', 60)
        y = height - ((pitch - min_pitch) / pitch_range * (height - 20) + 10)
        
        # Color by Sargam note
        sargam = note.get('sargam', 'Sa')
        colors = {
            'Sa': '#4ade80', 'Re': '#60a5fa', 'Ga': '#f472b6',
            'Ma': '#fbbf24', 'Pa': '#a78bfa', 'Dha': '#f87171',
            'Ni': '#2dd4bf', 'Sa_high': '#4ade80'
        }
        color = colors.get(sargam, '#888888')
        
        svg_notes.append(
            f'<rect x="{x}" y="{y-5}" width="6" height="10" '
            f'fill="{color}" rx="2" opacity="0.9"/>'
        )
    
    svg = f'''
    <svg width="{width}" height="{height}" style="background: rgba(20,20,40,0.8); border-radius: 8px;">
        <!-- Grid lines -->
        <line x1="0" y1="{height//2}" x2="{width}" y2="{height//2}" stroke="#333" stroke-dasharray="4"/>
        <!-- Notes -->
        {''.join(svg_notes)}
    </svg>
    '''
    
    st.markdown(f'<div class="piano-roll">{svg}</div>', unsafe_allow_html=True)
    
    if len(notes) > max_notes:
        st.caption(f"Showing first {max_notes} of {len(notes)} notes")


def create_export_zip(sequence: str, name: str, audio: np.ndarray, 
                       artwork, stats: dict) -> bytes:
    """Create ZIP file with all exports."""
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Audio WAV
        audio_buffer = io.BytesIO()
        import wave
        with wave.open(audio_buffer, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())
        audio_buffer.seek(0)
        zf.writestr(f"{name}_music.wav", audio_buffer.read())
        
        # Artwork PNG
        img_buffer = io.BytesIO()
        artwork.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        zf.writestr(f"{name}_art.png", img_buffer.read())
        
        # Stats report
        report = f"""# Genome Analysis Report: {name}

## Sequence Statistics
- Length: {stats['length']:,} bp
- GC Content: {stats['gc_content']:.1f}%
- Shannon Entropy: {stats['entropy']:.3f} bits
- N (Unknown): {stats['n_percent']:.1f}%

## Base Frequencies
- A: {stats['base_frequencies']['A']*100:.1f}%
- T: {stats['base_frequencies']['T']*100:.1f}%
- G: {stats['base_frequencies']['G']*100:.1f}%
- C: {stats['base_frequencies']['C']*100:.1f}%

## Repeat Analysis
- Tandem Repeats Found: {stats['repeat_count']}
- Repeat Density: {stats['repeat_density']*100:.1f}%

Generated by SynthOmics
"""
        zf.writestr(f"{name}_report.md", report)
        
        # Sequence FASTA
        fasta = f">{name}\n{sequence}\n"
        zf.writestr(f"{name}.fasta", fasta)
    
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("# 🧬 SynthOmics")
    st.markdown("---")
    
    # Input Section
    st.markdown("## 📂 Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Demo Sequences", "Upload FASTA", "Type Sequence"],
        horizontal=True
    )
    
    if input_method == "Demo Sequences":
        st.markdown("### 🧬 Disease demos (Normal vs Disease)")
        disease_pairs = {
            "Huntington (HTT): CAG20 vs CAG45": ("HTT Normal (CAG20)", "HTT Huntington-like (CAG45)"),
            "Myotonic dystrophy (DMPK): CTG12 vs CTG80": ("DMPK Normal (CTG12)", "DMPK DM1-like (CTG80)"),
            "Fragile X (FMR1): CGG30 vs CGG120": ("FMR1 Normal (CGG30)", "FMR1 Expanded (CGG120)"),
            "Sickle cell (HBB): normal vs point mutation": ("HBB Normal (HbA)", "HBB Sickle (HbS)")
        }
        selected_pair = st.selectbox("Select a disease pair:", list(disease_pairs.keys()))
        col_btn1, col_btn2 = st.columns(2)
        norm_demo, dis_demo = disease_pairs[selected_pair]
        with col_btn1:
            if st.button("Load Normal", use_container_width=True):
                result = gs_io.load_demo_sequence(norm_demo)
                if result:
                    header, seq = result
                    is_valid, cleaned, msg = gs_io.validate_sequence(seq)
                    if is_valid:
                        st.session_state.sequence = cleaned
                        st.session_state.sequence_name = norm_demo
                        st.success(f"Loaded Normal")
                        st.rerun()
        with col_btn2:
            if st.button("Load Disease", use_container_width=True):
                result = gs_io.load_demo_sequence(dis_demo)
                if result:
                    header, seq = result
                    is_valid, cleaned, msg = gs_io.validate_sequence(seq)
                    if is_valid:
                        st.session_state.sequence = cleaned
                        st.session_state.sequence_name = dis_demo
                        st.success(f"Loaded Disease")
                        st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Other Demos")
        demos = gs_io.get_demo_list()
        demo_names = [d['name'] for d in demos]
        
        selected_demo = st.selectbox(
            "Select demo sequence:",
            demo_names,
            index=0
        )
        
        demo_info = next(d for d in demos if d['name'] == selected_demo)
        st.caption(demo_info['description'])
        
        if st.button("🔬 Load Demo", use_container_width=True):
            result = gs_io.load_demo_sequence(selected_demo)
            if result:
                header, seq = result
                is_valid, cleaned, msg = gs_io.validate_sequence(seq)
                if is_valid:
                    st.session_state.sequence = cleaned
                    st.session_state.sequence_name = selected_demo
                    st.success(f"Loaded: {len(cleaned):,} bp")
    
    elif input_method == "Type Sequence":
        st.markdown("### ✍️ Manual Entry")
        
        if 'manual_seq_input' not in st.session_state:
            st.session_state.manual_seq_input = ""

        # Helper buttons for quick fill
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("ATG start example"):
                st.session_state.manual_seq_input = "ATGAAACCCGGGTTTAA"
        with col2:
            if st.button("CpG island example"):
                st.session_state.manual_seq_input = "CGCGCGCGCGCGCGCGCG"
        with col3:
            if st.button("Repeat example"):
                st.session_state.manual_seq_input = "ATATATATATATATAT"

        user_seq = st.text_area(
            "Paste or type DNA sequence (A/T/G/C only)",
            key="manual_seq_input",
            height=150,
            placeholder="Example: ATGCGTACGTTAG..."
        )
        
        st.caption("Tip: Try short genes, repeats, or CpG-rich sequences.")
        
        if st.button("🔬 Load Sequence", key="load_manual", use_container_width=True, type="primary"):
            if not user_seq:
                st.warning("Please enter a sequence.")
            else:
                try:
                    seq = gs_io.validate_dna_sequence(user_seq)
                    
                    # Performance check
                    if len(seq) > 20000:
                        st.info("Long sequence detected — sampling for real-time playback.")
                        seq = gs_io.sample_sequence(seq, 20000)

                    st.session_state.sequence = seq
                    st.session_state.sequence_name = "Manual Sequence"
                    st.session_state.audio = None
                    st.session_state.artwork = None
                    st.session_state.stats = None
                    st.session_state.notes = None
                    st.success(f"✅ Loaded: {len(seq):,} bp")
                    st.rerun()

                except ValueError as e:
                    st.error(str(e))
        
        st.info("Manual input allows quick exploration of motifs, mutations, CpG islands, and coding signals.")

    else:
        uploaded = st.file_uploader(
            "Upload FASTA file",
            type=['fa', 'fasta', 'fna'],
            help="FASTA format with .fa, .fasta, or .fna extension"
        )
        
        if uploaded:
            content = uploaded.read().decode('utf-8')
            sequences = gs_io.parse_fasta(content)
            
            if sequences:
                # Store parsed sequences in session state
                st.session_state['uploaded_sequences'] = sequences
                st.session_state['uploaded_filename'] = uploaded.name
        
        # Show sequence selector if we have uploaded sequences
        if 'uploaded_sequences' in st.session_state and st.session_state['uploaded_sequences']:
            sequences = st.session_state['uploaded_sequences']
            filename = st.session_state.get('uploaded_filename', 'uploaded')
            
            if len(sequences) > 1:
                seq_options = [f"{h[:30]}..." if len(h) > 30 else h 
                               for h, s in sequences]
                selected_idx = st.selectbox(
                    "Select sequence:",
                    range(len(sequences)),
                    format_func=lambda i: seq_options[i]
                )
            else:
                selected_idx = 0
            
            header, seq = sequences[selected_idx]
            is_valid, cleaned, msg = gs_io.validate_sequence(seq)
            
            if is_valid:
                st.caption(f"📄 {header[:40]}... ({len(cleaned):,} bp)")
                
                if st.button("🔬 Load Sequence", use_container_width=True, type="primary"):
                    # Sample if too long
                    if len(cleaned) > 50000:
                        cleaned = gs_io.sample_sequence(cleaned, 50000)
                        st.warning(f"Sampled to 50,000 bp for performance")
                    
                    st.session_state.sequence = cleaned
                    st.session_state.sequence_name = header[:30] if header else filename
                    # Clear previous generation
                    st.session_state.audio = None
                    st.session_state.artwork = None
                    st.session_state.stats = None
                    st.session_state.notes = None
                    st.success(f"✅ Loaded: {len(cleaned):,} bp")
                    st.rerun()
            else:
                st.error(f"Invalid sequence: {msg}")
    
    st.markdown("---")
    
    # Music Settings
    st.markdown("## 🎵 Music Settings")
    
    # Initialize scale_type in session state
    if 'scale_type' not in st.session_state:
        st.session_state.scale_type = 'sargam'
    
    # Initialize THz parameters
    if 'thz_k' not in st.session_state:
        st.session_state.thz_k = 5
    if 'thz_alpha' not in st.session_state:
        st.session_state.thz_alpha = 0.35
    if 'thz_quantize' not in st.session_state:
        st.session_state.thz_quantize = False
        
    scale_options = ["sargam", "major", "minor", "thz_physics"]
    scale_labels = {
        "sargam": "🕉️ Indian Classical (Sargam)",
        "major": "🎼 Western Major Scale",
        "minor": "🎹 Western Minor Scale",
        "thz_physics": "🔬 Physics Mode (THz)"
    }
    
    scale_type = st.radio(
        "Music System",
        scale_options,
        index=scale_options.index(st.session_state.scale_type),
        format_func=lambda x: scale_labels[x],
        horizontal=True,
        help="Choose the musical scale and tuning system"
    )
    
    if scale_type != st.session_state.scale_type:
        st.session_state.scale_type = scale_type
        st.session_state.audio = None
        st.session_state.notes = None
        st.rerun()
    
    # Show THz Physics controls if that mode is selected
    if scale_type == 'thz_physics':
        st.info("🔬 **Physics Mode**: Uses THz absorption peaks from Yu et al. 2019")
        
        col1, col2 = st.columns(2)
        with col1:
            thz_k = st.slider("Partials (K)", 3, 8, st.session_state.thz_k,
                             help="Number of THz peaks to use per nucleotide")
            if thz_k != st.session_state.thz_k:
                st.session_state.thz_k = thz_k
                st.session_state.audio = None
        
        with col2:
            thz_alpha = st.slider("Compression (α)", 0.20, 0.60, st.session_state.thz_alpha, 0.05,
                                  help="Lower = more compressed frequency range")
            if thz_alpha != st.session_state.thz_alpha:
                st.session_state.thz_alpha = thz_alpha
                st.session_state.audio = None
        
        # Advanced Quantization Controls
        st.markdown("##### 🎹 Quantization Settings")
        
        # Initialize advanced quant state
        if 'quant_type' not in st.session_state: st.session_state.quant_type = 'nearest'
        if 'quant_strength' not in st.session_state: st.session_state.quant_strength = 0.7
        if 'quant_smoothness' not in st.session_state: st.session_state.quant_smoothness = 0.6
        if 'quant_policy' not in st.session_state: st.session_state.quant_policy = 'fundamental_only'
        
        thz_quantize = st.toggle("Enable Quantization", st.session_state.thz_quantize)
        if thz_quantize != st.session_state.thz_quantize:
            st.session_state.thz_quantize = thz_quantize
            st.session_state.audio = None
            st.rerun()
            
        if thz_quantize:
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                quant_type = st.selectbox(
                    "Method", 
                    ['nearest', 'soft', 'smart'],
                    index=['nearest', 'soft', 'smart'].index(st.session_state.quant_type),
                    help="Nearest: Hard snap | Soft: Blend | Smart: Continuity-aware"
                )
                if quant_type != st.session_state.quant_type:
                    st.session_state.quant_type = quant_type
                    st.session_state.audio = None
            
            with q_col2:
                quant_policy = st.selectbox(
                    "Partials Policy",
                    ['fundamental_only', 'all'],
                    index=['fundamental_only', 'all'].index(st.session_state.quant_policy),
                    format_func=lambda x: "Fundamental Only (Maintains Timbre)" if x == 'fundamental_only' else "Quantize All (Harsher)",
                    help="Fundamental Only preserves the exact THz spectral ratios"
                )
                if quant_policy != st.session_state.quant_policy:
                    st.session_state.quant_policy = quant_policy
                    st.session_state.audio = None

            # Dynamic sliders based on method
            if quant_type == 'soft':
                quant_strength = st.slider("Strength", 0.0, 1.0, st.session_state.quant_strength, 0.1,
                                         help="0.0 = input frequency, 1.0 = hard snap")
                if quant_strength != st.session_state.quant_strength:
                    st.session_state.quant_strength = quant_strength
                    st.session_state.audio = None
            
            elif quant_type == 'smart':
                quant_smoothness = st.slider("Smoothness (Cost)", 0.0, 1.0, st.session_state.quant_smoothness, 0.1,
                                           help="Higher = stronger preference for small intervals and continuity")
                if quant_smoothness != st.session_state.quant_smoothness:
                    st.session_state.quant_smoothness = quant_smoothness
                    st.session_state.audio = None

        with st.expander("📜 Citation & Disclaimer"):
            st.markdown("""
**Primary Source**: Yu et al., Sensors 2019, 19(5):1148

**Disclaimer**: Molecular vibrations occur at THz (10¹² Hz), far above 
human hearing. This mode maps *published THz absorption peak positions* 
into audible range via ratio-preserving compression. **We are not 
"hearing molecules"**—this is data sonification.
            """)
    elif scale_type == 'sargam':
        st.markdown("#### 🕉️ Classical Mode Settings")
        with st.expander("ℹ️ Mapping Rationale"):
            st.markdown("""
            **Chemical Class -> Pitch Hierarchy**
            - **G (Guanine)**: Purine (Larger) → **Sa (Root)**
            - **A (Adenine)**: Purine (Larger) → **Re (2nd)**
            - **T (Thymine)**: Pyrimidine (Smaller) → **Ga (3rd)**
            - **C (Cytosine)**: Pyrimidine (Smaller) → **Ma (4th)**
            
            *The larger, heavier purines anchor the scale, while pyrimidines provide melodic movement.*
            """)
            
        st.markdown("##### 🎼 Structural Overlays")
        
        # Init State
        if 'overlay_pa' not in st.session_state: st.session_state.overlay_pa = False
        if 'overlay_dha' not in st.session_state: st.session_state.overlay_dha = False
        if 'overlay_ni' not in st.session_state: st.session_state.overlay_ni = False
        if 'overlay_cpg' not in st.session_state: st.session_state.overlay_cpg = False
        
        if 'mix_pa' not in st.session_state: st.session_state.mix_pa = 0.25
        if 'mix_dha' not in st.session_state: st.session_state.mix_dha = 0.35
        if 'mix_ni' not in st.session_state: st.session_state.mix_ni = 0.25
        if 'mix_cpg' not in st.session_state: st.session_state.mix_cpg = 0.20
        
        c1, c2 = st.columns(2)
        with c1:
             # Pa
             overlay_pa = st.toggle("Coding Regions (Pa)", st.session_state.overlay_pa, help="Harmonic overlay in ORF regions")
             if overlay_pa != st.session_state.overlay_pa:
                 st.session_state.overlay_pa = overlay_pa
                 st.session_state.audio = None
                 st.rerun()
             if st.session_state.overlay_pa:
                 st.session_state.mix_pa = st.slider("Pa Mix", 0.0, 1.0, st.session_state.mix_pa, 0.05)
             
             # Ni
             overlay_ni = st.toggle("High Entropy (Ni)", st.session_state.overlay_ni, help="Tension in complex regions")
             if overlay_ni != st.session_state.overlay_ni:
                 st.session_state.overlay_ni = overlay_ni
                 st.session_state.audio = None
                 st.rerun()
             if st.session_state.overlay_ni:
                 st.session_state.mix_ni = st.slider("Ni Mix", 0.0, 1.0, st.session_state.mix_ni, 0.05)
                 
        with c2:
             # Dha
             overlay_dha = st.toggle("Repeats (Dha)", st.session_state.overlay_dha, help="Accent at start of repeats")
             if overlay_dha != st.session_state.overlay_dha:
                 st.session_state.overlay_dha = overlay_dha
                 st.session_state.audio = None
                 st.rerun()
             if st.session_state.overlay_dha:
                 st.session_state.mix_dha = st.slider("Dha Mix", 0.0, 1.0, st.session_state.mix_dha, 0.05)
                 
             # CpG
             overlay_cpg = st.toggle("CpG Sites (Ornament)", st.session_state.overlay_cpg, help="Epigenetic marker ornament")
             if overlay_cpg != st.session_state.overlay_cpg:
                 st.session_state.overlay_cpg = overlay_cpg
                 st.session_state.audio = None
                 st.rerun()
             if st.session_state.overlay_cpg:
                 st.session_state.mix_cpg = st.slider("CpG Intensity", 0.0, 1.0, st.session_state.mix_cpg, 0.05)

        


    sa_frequency = st.slider(
        "Base Frequency (Sa/Root)",
        min_value=200,
        max_value=500,
        value=st.session_state.sa_frequency,
        key="sa_slider",
        help="Base frequency of Sa or Root Note",
        on_change=lambda: setattr(st.session_state, 'audio', None)  # Clear audio on change
    )
    # Update session state
    st.session_state.sa_frequency = sa_frequency
    
    sa_presets = {
        "C4 (261 Hz)": 261,
        "D4 (293 Hz)": 293,
        "A3 (220 Hz)": 220,
        "G3 (196 Hz)": 196,
    }
    preset = st.selectbox("Quick presets:", list(sa_presets.keys()))
    if st.button("Apply Preset", use_container_width=True):
        st.session_state.sa_frequency = sa_presets[preset]
        # Clear old audio so user regenerates with new settings
        st.session_state.audio = None
        st.session_state.notes = None
        st.toast(f"🎵 Sa set to {sa_presets[preset]} Hz - click Generate to hear!")
        st.rerun()
    
    # Initialize timbre in session state
    if 'timbre' not in st.session_state:
        st.session_state.timbre = 'synth'
    if 'drone_enabled' not in st.session_state:
        st.session_state.drone_enabled = True
    
    timbre_options = ["sitar", "harmonium", "veena", "bells", "strings", "organ", "lofi", "synth"]
    timbre_display = {
        "sitar": "🎸 Sitar",
        "harmonium": "🎹 Harmonium", 
        "veena": "🎵 Veena",
        "bells": "🔔 Bells",
        "strings": "🎻 Strings",
        "organ": "⛪ Organ",
        "lofi": "🧢 Lofi Keys (Gen Z)",
        "synth": "⚡ Synth Lead"
    }
    
    # Find current index
    current_idx = timbre_options.index(st.session_state.timbre) if st.session_state.timbre in timbre_options else 0
    
    timbre = st.selectbox(
        "Instrument",
        timbre_options,
        index=current_idx,
        format_func=lambda x: timbre_display.get(x, x),
        help="Each instrument has unique character!"
    )
    
    # If timbre changed, update and clear audio
    if timbre != st.session_state.timbre:
        st.session_state.timbre = timbre
        st.session_state.audio = None
        st.session_state.notes = None
    
    timbre_descriptions = {
        "sitar": "Bright attack, buzzy sympathetic strings, vibrato",
        "harmonium": "Warm, reedy, organ-like with chorus",
        "veena": "Smooth, rich, singing quality with vibrato", 
        "bells": "Bright, shimmering, fast decay",
        "strings": "Lush ensemble, slow attack, cinematic",
        "organ": "Full, sustained, Leslie-like vibrato",
        "lofi": "Chill, detuned keys with tape wobble (Recommended for Lofi beats!)",
        "synth": "Sharp sawtooth lead, EDM style"
    }
    st.caption(timbre_descriptions[timbre])
    
    drone_enabled = st.checkbox(
        "Enable Backing Track (Drone/Pad/Beats)", 
        value=st.session_state.drone_enabled,
        help="Background drone (Indian) or Beat+Pad (Western)"
    )
    
    # If drone changed, update and clear audio
    if drone_enabled != st.session_state.drone_enabled:
        st.session_state.drone_enabled = drone_enabled
        st.session_state.audio = None
        st.session_state.notes = None
    
    st.markdown("---")
    
    # Art Settings
    st.markdown("## 🎨 Art Settings")
    
    art_style = st.selectbox(
        "Style",
        ["heatmap_walk", "chaos", "mosaic"],
        format_func=lambda x: {
            "heatmap_walk": "Heatmap + DNA Walk",
            "chaos": "Entropy Chaos",
            "mosaic": "Codon Mosaic"
        }[x]
    )
    
    art_size = st.slider("Image Size", 400, 1200, 800, 100)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 3em; margin-bottom: 10px;">🧬 SynthOmics</h1>
    <p style="color: #888; font-size: 1.2em;">
        Transform DNA sequences into deterministic Sargam music and generative art
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Generate", 
    "📖 Explain", 
    "🔬 Compare", 
    "📥 Export"
])

# =============================================================================
# TAB 1: GENERATE
# =============================================================================
with tab1:
    if st.session_state.sequence:
        seq = st.session_state.sequence
        name = st.session_state.sequence_name or "Unknown"
        
        # Sequence info bar
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 🧬 {name}")
        with col2:
            st.metric("Length", f"{len(seq):,} bp")
        with col3:
            if st.button("🎬 Generate All", type="primary", use_container_width=True):
                with st.spinner("Generating music and art..."):
                    # Calculate stats
                    st.session_state.stats = gs_stats.get_sequence_stats(seq)
                    
                    # Generate music (with THz params if applicable)
                    audio, notes = gs_music.generate_music(
                        seq,
                        sa_frequency=float(sa_frequency),
                        timbre=timbre,
                        drone_enabled=drone_enabled,
                        gc_content=st.session_state.stats['gc_content'],
                        entropy=st.session_state.stats['entropy'],
                        scale_type=st.session_state.scale_type,
                        thz_k=st.session_state.thz_k,
                        thz_alpha=st.session_state.thz_alpha,
                        thz_mode='lowest',
                        thz_quantize=st.session_state.thz_quantize,
                        quant_type=st.session_state.get('quant_type', 'nearest'),
                        quant_strength=st.session_state.get('quant_strength', 0.7),
                        quant_smoothness=st.session_state.get('quant_smoothness', 0.6),
                        quant_policy=st.session_state.get('quant_policy', 'fundamental_only'),
                        overlay_pa=st.session_state.get('overlay_pa', False),
                        overlay_dha=st.session_state.get('overlay_dha', False),
                        overlay_ni=st.session_state.get('overlay_ni', False),
                        overlay_cpg=st.session_state.get('overlay_cpg', False),
                        mix_pa=st.session_state.get('mix_pa', 0.25),
                        mix_dha=st.session_state.get('mix_dha', 0.35),
                        mix_ni=st.session_state.get('mix_ni', 0.25),
                        mix_cpg=st.session_state.get('mix_cpg', 0.20)
                    )
                    st.session_state.audio = audio
                    st.session_state.notes = notes
                    
                    # Generate art
                    st.session_state.artwork = gs_art.generate_art(
                        seq, 
                        style=art_style,
                        width=art_size,
                        height=art_size
                    )
                
                st.success("✅ Generated!")
                st.rerun()
        
        # Display results
        if st.session_state.audio is not None:
            col_music, col_art = st.columns([1, 1])
            
            with col_music:
                st.markdown("### 🎵 Music")
                
                # Audio player
                audio_b64 = audio_to_base64(st.session_state.audio)
                components.html(gs_viz.get_wavesurfer_html(audio_b64), height=150)
                
                # Piano roll
                st.markdown("#### Piano Roll")
                if st.session_state.notes:
                    # Add pitch to notes for visualization
                    for note in st.session_state.notes:
                        note['pitch'] = gs_music.sargam_to_midi_note(note['sargam'])
                    render_piano_roll(st.session_state.notes)
                
                # Music stats
                with st.expander("🎼 Music Details"):
                    stats = st.session_state.stats
                    st.markdown(f"""
                    - **Tempo factor**: {0.5 + stats['gc_content']/100 * 0.8:.2f}x
                    - **Total notes**: {len(st.session_state.notes):,}
                    - **Duration**: ~{len(st.session_state.audio)/44100:.1f}s
                    - **Sa frequency**: {sa_frequency} Hz
                    - **Timbre**: {timbre}
                    """)
            
            with col_art:
                st.markdown("### 🎨 Art")
                
                # Display artwork
                st.image(st.session_state.artwork, use_container_width=True)
                
                # Hint to export tab (no auto-download here)
                st.caption("💡 Go to **Export** tab to download files")
                
            # --- AUDIO DIAGNOSTICS ---
            with st.expander("🛠️ Audio Diagnostics"):
                import io
                import matplotlib.pyplot as plt
                
                # Read WAV for analysis
                audio_buffer = io.BytesIO(st.session_state.audio)
                try:
                    sr, samples = gs_viz.read_wav(audio_buffer)
                    
                    st.markdown("#### Waveform")
                    fig_wave = gs_viz.plot_waveform(samples, sr)
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)
                    
                    st.markdown("#### RMS Energy")
                    fig_rms = gs_viz.plot_rms_energy(samples, sr)
                    st.pyplot(fig_rms)
                    plt.close(fig_rms)
                    
                    st.markdown("#### Spectrogram")
                    fig_spec = gs_viz.plot_spectrogram(samples, sr)
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)
                except Exception as e:
                    st.error(f"Cannot generate diagnostics: {e}")
        
        # Stats panel
        if st.session_state.stats:
            st.markdown("---")
            st.markdown("### 📊 Sequence Statistics")
            
            stats = st.session_state.stats
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("GC Content", f"{stats['gc_content']:.1f}%")
            with col2:
                st.metric("Entropy", f"{stats['entropy']:.3f} bits")
            with col3:
                st.metric("Repeats", stats['repeat_count'])
            with col4:
                st.metric("N%", f"{stats['n_percent']:.1f}%")
            
            # Base composition chart
            with st.expander("📊 Base Composition"):
                import pandas as pd
                base_data = pd.DataFrame({
                    'Base': ['A', 'T', 'G', 'C'],
                    'Frequency': [
                        stats['base_frequencies']['A'],
                        stats['base_frequencies']['T'],
                        stats['base_frequencies']['G'],
                        stats['base_frequencies']['C']
                    ]
                })
                st.bar_chart(base_data.set_index('Base'))
            
            # --- MUTATION PLAYGROUND ---
            st.markdown("---")
            st.subheader("🧬 Mutation Playground")
            
            st.info("Single nucleotide mutations can alter protein structure, gene regulation, or repeat stability. This playground demonstrates how small genomic changes alter pattern and sound.")
            
            mut_type = st.radio(
                "Choose mutation type:",
                ["Single base substitution", "Random SNP", "Insertion", "Deletion", "Repeat expansion"],
                horizontal=True
            )
            
            mutated_seq = None
            diff_html = ""
            
            if mut_type == "Single base substitution":
                col1, col2 = st.columns(2)
                with col1:
                    pos = st.slider("Position", 1, len(seq), 1)
                with col2:
                    current_base = seq[pos-1]
                    bases = ["A", "T", "G", "C"]
                    # Default to next base
                    default_idx = (bases.index(current_base) + 1) % 4 if current_base in bases else 0
                    new_base = st.selectbox("New base", bases, index=default_idx)
                
                mutated_seq = seq[:pos-1] + new_base + seq[pos:]
                
                # Display difference
                window = 20
                start_idx = max(0, pos-1 - window)
                end_idx = min(len(seq), pos + window)
                
                orig_context = seq[start_idx:pos-1] + f"<span style='color: white; background-color: #f43f5e; padding: 0 4px; border-radius: 4px; font-weight: bold;'>{current_base}</span>" + seq[pos:end_idx]
                mut_context = seq[start_idx:pos-1] + f"<span style='color: white; background-color: #10b981; padding: 0 4px; border-radius: 4px; font-weight: bold;'>{new_base}</span>" + seq[pos:end_idx]
                
                diff_html = f"""
                <div style='font-family: monospace; font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                    <div><b>Original:</b> ...{orig_context}...</div>
                    <div style='margin-top: 5px;'><b>Mutated:</b> ...{mut_context}...</div>
                </div>
                """
                
            elif mut_type == "Random SNP":
                if st.button("Generate Random Mutation"):
                    import random
                    pos = random.randint(1, len(seq))
                    current_base = seq[pos-1]
                    bases = ["A", "T", "G", "C"]
                    if current_base in bases:
                        bases.remove(current_base)
                    new_base = random.choice(bases)
                    
                    st.session_state.random_mut_pos = pos
                    st.session_state.random_mut_base = new_base
                
                if 'random_mut_pos' in st.session_state:
                    pos = st.session_state.random_mut_pos
                    new_base = st.session_state.random_mut_base
                    current_base = seq[pos-1]
                    
                    st.write(f"Mutated at position **{pos}**: {current_base} → {new_base}")
                    
                    mutated_seq = seq[:pos-1] + new_base + seq[pos:]
                    
                    window = 20
                    start_idx = max(0, pos-1 - window)
                    end_idx = min(len(seq), pos + window)
                    
                    orig_context = seq[start_idx:pos-1] + f"<span style='color: white; background-color: #f43f5e; padding: 0 4px; border-radius: 4px; font-weight: bold;'>{current_base}</span>" + seq[pos:end_idx]
                    mut_context = seq[start_idx:pos-1] + f"<span style='color: white; background-color: #10b981; padding: 0 4px; border-radius: 4px; font-weight: bold;'>{new_base}</span>" + seq[pos:end_idx]
                    
                    diff_html = f"""
                    <div style='font-family: monospace; font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                        <div><b>Original:</b> ...{orig_context}...</div>
                        <div style='margin-top: 5px;'><b>Mutated:</b> ...{mut_context}...</div>
                    </div>
                    """
            
            elif mut_type == "Insertion":
                col1, col2 = st.columns(2)
                with col1:
                    pos = st.slider("Position to insert AFTER", 0, len(seq), 1)
                with col2:
                    new_base = st.selectbox("Base to insert", ["A", "T", "G", "C"])
                
                mutated_seq = seq[:pos] + new_base + seq[pos:]
                
                window = 20
                start_idx = max(0, pos - window)
                end_idx = min(len(seq), pos + window)
                
                # Show insertion point in original
                orig_context = seq[start_idx:pos] + f"<span style='color: #f59e0b; font-weight: bold;'>|</span>" + seq[pos:end_idx]
                mut_context = seq[start_idx:pos] + f"<span style='color: white; background-color: #3b82f6; padding: 0 4px; border-radius: 4px; font-weight: bold;'>{new_base}</span>" + seq[pos:end_idx]
                
                diff_html = f"""
                <div style='font-family: monospace; font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                    <div><b>Original:</b> ...{orig_context}...</div>
                    <div style='margin-top: 5px;'><b>Mutated:</b> ...{mut_context}...</div>
                </div>
                """
                
            elif mut_type == "Deletion":
                pos = st.slider("Position to delete", 1, len(seq), 1)
                
                current_base = seq[pos-1]
                mutated_seq = seq[:pos-1] + seq[pos:]
                
                window = 20
                start_idx = max(0, pos-1 - window)
                end_idx = min(len(seq), pos + window)
                
                orig_context = seq[start_idx:pos-1] + f"<span style='color: #94a3b8; text-decoration: line-through; padding: 0 4px; font-weight: bold;'>{current_base}</span>" + seq[pos:end_idx]
                mut_context = seq[start_idx:pos-1] + f"<span style='color: #f59e0b; font-weight: bold;'>_</span>" + seq[pos:end_idx]
                
                diff_html = f"""
                <div style='font-family: monospace; font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                    <div><b>Original:</b> ...{orig_context}...</div>
                    <div style='margin-top: 5px;'><b>Mutated:</b> ...{mut_context}...</div>
                </div>
                """
                
            elif mut_type == "Repeat expansion":
                col1, col2, col3 = st.columns(3)
                with col1:
                    motif = st.text_input("Motif to repeat (2-4 bp)", value="CAG", max_chars=4).upper()
                    # Filter invalid chars
                    motif = ''.join(c for c in motif if c in "ATGC")
                    if not motif: motif = "CAG"
                with col2:
                    pos = st.slider("Position to insert at", 1, len(seq), max(1, len(seq)//2))
                with col3:
                    repeats = st.slider("Number of repeats", 3, 10, 5)
                
                added_seq = motif * repeats
                mutated_seq = seq[:pos-1] + added_seq + seq[pos-1:]
                
                window = 10
                start_idx = max(0, pos-1 - window)
                end_idx = min(len(seq), pos-1 + window)
                
                # Show insertion point in original
                orig_context = seq[start_idx:pos-1] + f"<span style='color: #f59e0b; font-weight: bold;'>|</span>" + seq[pos-1:end_idx]
                
                # Format repeats with alternating colors for readability
                formatted_repeats = ""
                for i in range(repeats):
                    bg_color = "#8b5cf6" if i % 2 == 0 else "#a78bfa"
                    formatted_repeats += f"<span style='color: white; background-color: {bg_color}; padding: 0 2px; border-radius: 2px;'>{motif}</span>"
                
                mut_context = seq[start_idx:pos-1] + f"<span style='margin: 0 2px;'>{formatted_repeats}</span>" + seq[pos-1:end_idx]
                
                diff_html = f"""
                <div style='font-family: monospace; font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; overflow-x: auto;'>
                    <div style='white-space: nowrap;'><b>Original:</b> ...{orig_context}...</div>
                    <div style='margin-top: 5px; white-space: nowrap;'><b>Mutated:</b> ...{mut_context}...</div>
                </div>
                """

            if diff_html:
                st.markdown(diff_html, unsafe_allow_html=True)
            
            if mutated_seq:
                st.markdown("#### 🎧 Hear the difference")
                
                if st.button("Generate Comparison Audio", type="primary"):
                    with st.spinner("Generating audio for both sequences..."):
                        # Ensure we get fresh logic
                        st.session_state.stats = gs_stats.get_sequence_stats(seq)
                        
                        orig_audio, orig_notes = gs_music.generate_music(
                            seq,
                            sa_frequency=float(sa_frequency),
                            timbre=timbre,
                            drone_enabled=drone_enabled,
                            gc_content=st.session_state.stats['gc_content'],
                            entropy=st.session_state.stats['entropy'],
                            scale_type=st.session_state.scale_type,
                            thz_k=st.session_state.thz_k,
                            thz_alpha=st.session_state.thz_alpha,
                            thz_mode='lowest',
                            thz_quantize=st.session_state.thz_quantize,
                            quant_type=st.session_state.get('quant_type', 'nearest'),
                            quant_strength=st.session_state.get('quant_strength', 0.7),
                            quant_smoothness=st.session_state.get('quant_smoothness', 0.6),
                            quant_policy=st.session_state.get('quant_policy', 'fundamental_only'),
                            overlay_pa=st.session_state.get('overlay_pa', False),
                            overlay_dha=st.session_state.get('overlay_dha', False),
                            overlay_ni=st.session_state.get('overlay_ni', False),
                            overlay_cpg=st.session_state.get('overlay_cpg', False),
                            mix_pa=st.session_state.get('mix_pa', 0.25),
                            mix_dha=st.session_state.get('mix_dha', 0.35),
                            mix_ni=st.session_state.get('mix_ni', 0.25),
                            mix_cpg=st.session_state.get('mix_cpg', 0.20)
                        )
                        
                        mutated_stats = gs_stats.get_sequence_stats(mutated_seq)
                        mut_audio, mut_notes = gs_music.generate_music(
                            mutated_seq,
                            sa_frequency=float(sa_frequency),
                            timbre=timbre,
                            drone_enabled=drone_enabled,
                            gc_content=mutated_stats['gc_content'],
                            entropy=mutated_stats['entropy'],
                            scale_type=st.session_state.scale_type,
                            thz_k=st.session_state.thz_k,
                            thz_alpha=st.session_state.thz_alpha,
                            thz_mode='lowest',
                            thz_quantize=st.session_state.thz_quantize,
                            quant_type=st.session_state.get('quant_type', 'nearest'),
                            quant_strength=st.session_state.get('quant_strength', 0.7),
                            quant_smoothness=st.session_state.get('quant_smoothness', 0.6),
                            quant_policy=st.session_state.get('quant_policy', 'fundamental_only'),
                            overlay_pa=st.session_state.get('overlay_pa', False),
                            overlay_dha=st.session_state.get('overlay_dha', False),
                            overlay_ni=st.session_state.get('overlay_ni', False),
                            overlay_cpg=st.session_state.get('overlay_cpg', False),
                            mix_pa=st.session_state.get('mix_pa', 0.25),
                            mix_dha=st.session_state.get('mix_dha', 0.35),
                            mix_ni=st.session_state.get('mix_ni', 0.25),
                            mix_cpg=st.session_state.get('mix_cpg', 0.20)
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Sequence**")
                            orig_b64 = audio_to_base64(orig_audio)
                            components.html(gs_viz.get_wavesurfer_html(orig_b64), height=150)
                            
                        with col2:
                            st.markdown("**Mutated Sequence**")
                            mut_b64 = audio_to_base64(mut_audio)
                            components.html(gs_viz.get_wavesurfer_html(mut_b64), height=150)
    
    else:
        st.info("👈 Select a demo sequence or upload a FASTA file to get started!")
        
        # Quick demo buttons
        st.markdown("### Quick Start")
        cols = st.columns(5)
        demos = gs_io.get_demo_list()[:5]
        
        for i, demo in enumerate(demos):
            with cols[i]:
                if st.button(demo['name'], use_container_width=True):
                    result = gs_io.load_demo_sequence(demo['name'])
                    if result:
                        header, seq = result
                        is_valid, cleaned, msg = gs_io.validate_sequence(seq)
                        if is_valid:
                            st.session_state.sequence = cleaned
                            st.session_state.sequence_name = demo['name']
                            st.rerun()


# =============================================================================
# TAB 2: EXPLAIN
# =============================================================================
with tab2:
    st.markdown("## How It Works")
    
    st.markdown("""
    <div class="info-box">
    <strong>Core Concept:</strong> DNA is information. We translate this information into 
    sound and image using deterministic, scientifically-grounded mappings.
    Each genome produces a unique but reproducible sensory signature.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎵 DNA → Music Mapping")
        st.markdown(gs_music.get_mapping_legend())
    
    with col2:
        st.markdown("### 🎨 DNA → Art Mapping")
        st.markdown(gs_art.get_art_mapping_legend())
    
    st.markdown("---")
    
    st.markdown("### 🔬 Scientific Principles")
    
    with st.expander("Information Theory"):
        st.markdown("""
        **Shannon Entropy** measures the information content (randomness) of a sequence:
        - Low entropy (close to 0): Highly repetitive, predictable
        - High entropy (close to 2 for DNA): Random, high information density
        
        We use entropy to modulate note density and dynamics—random sequences 
        sound more active and louder.
        """)
    
    with st.expander("Sargam System"):
        st.markdown("""
        **Sargam** is the Indian classical music solfège (like Western Do-Re-Mi):
        
        | Sargam | Western | Ratio |
        |--------|---------|-------|
        | Sa | Do (Tonic) | 1:1 |
        | Re | Re | 9:8 |
        | Ga | Mi | 5:4 |
        | Ma | Fa | 4:3 |
        | Pa | Sol | 3:2 |
        | Dha | La | 5:3 |
        | Ni | Ti | 15:8 |
        
        The **movable Sa** allows shifting the entire scale to different frequencies,
        changing the emotional character while preserving intervals.
        """)
        
    with st.expander("Sargam Quantization (Physics Mode)"):
        st.markdown("""
        **Sargam quantization** is applied in log-frequency space relative to Sa. A **continuity-aware cost function** 
        avoids unnatural jumps and ensures musical smoothness.
        
        - **Smart Quantization**: Uses Viterbi-like logic to balance pitch accuracy with melodic continuity.
        - **Soft Quantization**: Blends the raw THz frequency with the snapped note, creating a microtonal effect.
        - **Partial Policy**: In Physics Mode, we typically quantize only the **fundamental** frequency and preserve 
          the ratios of the upper partials. This maintains the unique **THz-derived timbre** while making the melody 
          consonant with the musical scale.
        """)
    
    with st.expander("K-mer Analysis"):
        st.markdown("""
        **K-mers** are subsequences of length k. For k=4, there are 4^4 = 256 possible 4-mers.
        
        The frequency distribution of k-mers creates a "fingerprint" of the genome:
        - Viruses often have skewed k-mer profiles
        - Different species have distinct k-mer signatures
        - We use this to create heatmap textures in the art
        """)
    
    with st.expander("DNA Walk"):
        st.markdown("""
        The **DNA walk** maps sequence to a 2D trajectory:
        - A → Up
        - T → Down  
        - G → Right
        - C → Left
        
        This reveals:
        - GC/AT skew (drift in the walk)
        - Compositional structure
        - Repeat regions (loops in the path)
        """)
        
    with st.expander("Scientific References"):
        st.markdown("""
        - **Huntington's Disease (HTT)**: CAG repeat expansion ≥36 is associated with Huntington's disease ([Stine et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7990409/)).
        - **Myotonic Dystrophy (DMPK)**: Normal CTG repeats ~5–35; pathogenic expansions >50 ([Brook et al. / Nature](https://www.nature.com/)).
        - **Fragile X Syndrome (FMR1)**: Full mutation involves >200 CGG repeats ([Loomis et al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3565225/)).
        - **Sickle Cell Disease (HBB)**: Codon 6 point mutation GAG → GTG changes Glutamate to Valine, causing HbS ([Rees et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5746148/)).
        """)


# =============================================================================
# TAB 3: COMPARE
# =============================================================================
with tab3:
    st.markdown("## Compare Two Genomes")
    
    st.markdown("### 🧬 Quick Load: Disease Pairs")
    disease_pairs = {
        "Huntington (HTT): CAG20 vs CAG45": ("HTT Normal (CAG20)", "HTT Huntington-like (CAG45)"),
        "Myotonic dystrophy (DMPK): CTG12 vs CTG80": ("DMPK Normal (CTG12)", "DMPK DM1-like (CTG80)"),
        "Fragile X (FMR1): CGG30 vs CGG120": ("FMR1 Normal (CGG30)", "FMR1 Expanded (CGG120)"),
        "Sickle cell (HBB): normal vs point mutation": ("HBB Normal (HbA)", "HBB Sickle (HbS)")
    }
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        sel_pair = st.selectbox("Load demo pair:", list(disease_pairs.keys()))
    with col_p2:
        if st.button("Load Pair Into Compare", use_container_width=True):
            n_name, d_name = disease_pairs[sel_pair]
            r1 = gs_io.load_demo_sequence(n_name)
            r2 = gs_io.load_demo_sequence(d_name)
            if r1 and r2:
                _, seq1 = r1
                _, seq2 = r2
                _, cl1, _ = gs_io.validate_sequence(seq1)
                _, cl2, _ = gs_io.validate_sequence(seq2)
                st.session_state.seq1 = cl1
                st.session_state.seq1_name = n_name
                st.session_state.seq2 = cl2
                st.session_state.seq2_name = d_name
                st.rerun()
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sequence 1")
        
        # Demo or current sequence
        demos = gs_io.get_demo_list()
        options1 = ["(Current sequence)"] + [d['name'] for d in demos]
        
        choice1 = st.selectbox("Select:", options1, key="cmp1")
        
        if choice1 == "(Current sequence)" and st.session_state.sequence:
            st.session_state.seq1 = st.session_state.sequence
            st.session_state.seq1_name = st.session_state.sequence_name
        elif choice1 != "(Current sequence)":
            result = gs_io.load_demo_sequence(choice1)
            if result:
                _, seq = result
                _, cleaned, _ = gs_io.validate_sequence(seq)
                st.session_state.seq1 = cleaned
                st.session_state.seq1_name = choice1
        
        if st.session_state.seq1:
            st.success(f"✓ {st.session_state.seq1_name}: {len(st.session_state.seq1):,} bp")
    
    with col2:
        st.markdown("### Sequence 2")
        
        options2 = [d['name'] for d in demos]
        choice2 = st.selectbox("Select:", options2, key="cmp2", index=1)
        
        result = gs_io.load_demo_sequence(choice2)
        if result:
            _, seq = result
            _, cleaned, _ = gs_io.validate_sequence(seq)
            st.session_state.seq2 = cleaned
            st.session_state.seq2_name = choice2
        
        if st.session_state.seq2:
            st.success(f"✓ {st.session_state.seq2_name}: {len(st.session_state.seq2):,} bp")
    
    st.markdown("---")
    
    if st.session_state.seq1 and st.session_state.seq2:
        st.markdown("### 🎛️ Difference Sonification Settings")
        col_chk1, col_chk2, col_chk3 = st.columns(3)
        with col_chk1:
            do_glitch = st.checkbox("Highlight mutations (glitch layer)", value=True)
        with col_chk2:
            do_pulse = st.checkbox("Highlight repeats (pulse layer)", value=True)
        with col_chk3:
            gen_diff = st.checkbox("Generate difference track", value=True)

        if st.button("🔬 Compare", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                comparison = gs_compare.compare_sequences(
                    st.session_state.seq1,
                    st.session_state.seq2,
                    st.session_state.seq1_name,
                    st.session_state.seq2_name
                )
            
            # Display comparison
            st.markdown("### Results")
            
            # Metrics
            sim = comparison['similarity']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cosine Similarity", f"{sim['cosine_similarity']:.3f}")
            with col2:
                st.metric("Jensen-Shannon Div.", f"{sim['jensen_shannon_divergence']:.3f}")
            with col3:
                st.metric("Base Composition", f"{sim['base_composition_similarity']:.3f}")
            
            # Interpretation
            st.markdown(comparison['interpretation'])
            
            # Side by side stats
            st.markdown("### Statistics Comparison")
            s1 = comparison['sequence1']['stats']
            s2 = comparison['sequence2']['stats']
            
            comparison_df = {
                'Metric': ['Length (bp)', 'GC Content (%)', 'Entropy (bits)', 'Repeats'],
                comparison['sequence1']['name']: [
                    f"{s1['length']:,}",
                    f"{s1['gc_content']:.1f}",
                    f"{s1['entropy']:.3f}",
                    str(s1['repeat_count'])
                ],
                comparison['sequence2']['name']: [
                    f"{s2['length']:,}",
                    f"{s2['gc_content']:.1f}",
                    f"{s2['entropy']:.3f}",
                    str(s2['repeat_count'])
                ]
            }
            st.table(comparison_df)
            
            # Side by side art
            st.markdown("### Visual Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                art1 = gs_art.generate_art(st.session_state.seq1, style='chaos', width=400, height=400)
                st.image(art1, caption=st.session_state.seq1_name)
            
            with col2:
                art2 = gs_art.generate_art(st.session_state.seq2, style='chaos', width=400, height=400)
                st.image(art2, caption=st.session_state.seq2_name)
                
            # --- Disease Demo Specific Analysis ---
            if st.session_state.seq1_name and st.session_state.seq2_name:
                s1_name_up = st.session_state.seq1_name.upper()
                s2_name_up = st.session_state.seq2_name.upper()
                is_repeat = any(motif in s1_name_up or motif in s2_name_up for motif in ["CAG", "CTG", "CGG"])
                is_hbb = "HBB" in s1_name_up or "HBB" in s2_name_up
                
                if is_repeat:
                    st.markdown("---")
                    st.markdown("### 🧬 Repeat Expansion Analysis")
                    motif = "CAG"
                    if "CTG" in s1_name_up or "CTG" in s2_name_up: motif = "CTG"
                    if "CGG" in s1_name_up or "CGG" in s2_name_up: motif = "CGG"
                    
                    def get_max_repeat(s, m):
                        import re
                        matches = re.findall(f"((?:{m})+)", s)
                        return max(len(match) // len(m) for match in matches) if matches else 0
                    
                    r1 = get_max_repeat(st.session_state.seq1, motif)
                    r2 = get_max_repeat(st.session_state.seq2, motif)
                    
                    st.info(f"Detected **{motif}** repeat expansion demo.")
                    col_r1, col_r2 = st.columns(2)
                    col_r1.metric(f"{st.session_state.seq1_name} ({motif} repeats)", r1)
                    col_r2.metric(f"{st.session_state.seq2_name} ({motif} repeats)", r2)
                
                elif is_hbb:
                    st.markdown("---")
                    st.markdown("### 🧬 Point Mutation Analysis (HBB)")
                    s1 = st.session_state.seq1
                    s2 = st.session_state.seq2
                    mismatches = [i for i, (b1, b2) in enumerate(zip(s1, s2)) if b1 != b2]
                    
                    if mismatches:
                        st.info("Detected Point Mutation(s)")
                        for m in mismatches:
                            codon_start = (m // 3) * 3
                            c1 = s1[codon_start:codon_start+3]
                            c2 = s2[codon_start:codon_start+3]
                            st.write(f"Mismatch at position {m+1}:")
                            st.code(f"{st.session_state.seq1_name} codon: {c1}\n{st.session_state.seq2_name} codon: {c2}")
                            
            if gen_diff:
                st.markdown("---")
                st.markdown("### 🎧 Playable Audio Tracks")
                
                with st.spinner("Synthesizing audio comparison..."):
                    # Generate base audio 1 & 2
                    synth = gs_music.GenomeSynthesizer()
                    _, a1 = synth.sequence_to_audio(
                        st.session_state.seq1, 
                        base_duration=0.125, 
                        gc_content=comparison['sequence1']['stats']['gc_content'],
                        entropy=comparison['sequence1']['stats']['entropy']
                    )
                    _, a2 = synth.sequence_to_audio(
                        st.session_state.seq2, 
                        base_duration=0.125, 
                        gc_content=comparison['sequence2']['stats']['gc_content'],
                        entropy=comparison['sequence2']['stats']['entropy']
                    )
                    
                    diff_buffer = gs_diff.generate_difference_track(
                        st.session_state.seq1, 
                        st.session_state.seq2, 
                        a2, 
                        note_duration=0.125,
                        do_glitch=do_glitch,
                        do_pulse=do_pulse
                    )
                    
                col_aud1, col_aud2, col_aud3 = st.columns(3)
                
                with col_aud1:
                    st.markdown(f"**{st.session_state.seq1_name}**")
                    b1 = audio_to_base64(a1)
                    components.html(gs_viz.get_wavesurfer_html(b1), height=150)
                
                with col_aud2:
                    st.markdown(f"**{st.session_state.seq2_name}**")
                    b2 = audio_to_base64(a2)
                    components.html(gs_viz.get_wavesurfer_html(b2), height=150)
                
                with col_aud3:
                    st.markdown("**Difference Track**")
                    diff_b64 = base64.b64encode(diff_buffer.read()).decode()
                    diff_buffer.seek(0)
                    components.html(gs_viz.get_wavesurfer_html(diff_b64), height=150)
                    
                with st.expander("🛠️ Difference Track Diagnostics"):
                    import matplotlib.pyplot as plt
                    sr, samples = gs_viz.read_wav(diff_buffer)
                    _sr, base_samples = gs_viz.read_wav(io.BytesIO(a2))
                    
                    st.markdown("#### Waveform Comparison")
                    fig_cmp = gs_viz.plot_waveform_comparison(
                        base_samples, 
                        samples, 
                        sr, 
                        title1=st.session_state.seq2_name + " (Base)", 
                        title2="Differences",
                        title="Waveform Overlays"
                    )
                    st.pyplot(fig_cmp)
                    plt.close(fig_cmp)
                    
                    st.markdown("#### Difference Track Spectrogram")
                    fig_spec = gs_viz.plot_spectrogram(samples, sr, title="Diff Track Spectrogram")
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)
    
    else:
        st.info("Select two sequences to compare")
    
    # Comparison legend
    with st.expander("📖 Understanding Comparison Metrics"):
        st.markdown(gs_compare.get_comparison_legend())


# =============================================================================
# TAB 4: EXPORT
# =============================================================================
with tab4:
    st.markdown("## Export Your Results")
    
    if st.session_state.audio is not None and st.session_state.artwork is not None:
        st.success("✅ Content ready for export!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Individual Files")
            
            # WAV
            audio_b64 = audio_to_base64(st.session_state.audio)
            st.download_button(
                "🎵 Download WAV",
                data=base64.b64decode(audio_b64),
                file_name=f"{st.session_state.sequence_name or 'genome'}_music.wav",
                mime="audio/wav",
                use_container_width=True,
                key="export_wav_btn"
            )
            
            # PNG
            img_b64 = image_to_base64(st.session_state.artwork)
            st.download_button(
                "🎨 Download PNG",
                data=base64.b64decode(img_b64),
                file_name=f"{st.session_state.sequence_name or 'genome'}_art.png",
                mime="image/png",
                use_container_width=True,
                key="export_png_btn"
            )
            
            # MIDI (if available)
            if gs_music.HAS_MIDIUTIL:
                midi_buffer = io.BytesIO()
                # Generate MIDI
                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                    gs_music.sequence_to_midi(st.session_state.sequence, tmp.name)
                    with open(tmp.name, 'rb') as f:
                        midi_data = f.read()
                    os.unlink(tmp.name)
                
                st.download_button(
                    "🎹 Download MIDI",
                    data=midi_data,
                    file_name=f"{st.session_state.sequence_name or 'genome'}_music.mid",
                    mime="audio/midi",
                    use_container_width=True,
                    key="export_midi_btn"
                )
        
        with col2:
            st.markdown("### Complete Package")
            
            # ZIP
            zip_data = create_export_zip(
                st.session_state.sequence,
                st.session_state.sequence_name or 'genome',
                st.session_state.audio,
                st.session_state.artwork,
                st.session_state.stats
            )
            
            st.download_button(
                "📦 Download ZIP (All Files)",
                data=zip_data,
                file_name=f"{st.session_state.sequence_name or 'genome'}_package.zip",
                mime="application/zip",
                use_container_width=True,
                key="export_zip_btn"
            )
            
            st.markdown("""
            **ZIP includes:**
            - Music (WAV)
            - Artwork (PNG)
            - Analysis report (Markdown)
            - Original sequence (FASTA)
            """)
    
    else:
        st.info("👈 Generate music and art first, then come here to export!")
    
    st.markdown("---")
    
    # Stats summary
    if st.session_state.stats:
        st.markdown("### 📊 Analysis Summary")
        st.markdown(gs_stats.format_stats_for_display(st.session_state.stats))


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🧬 <strong>SynthOmics</strong> | Science Day Demo</p>
    <p style="font-size: 0.9em;">
        DNA is information. We experience it through sound and sight.
    </p>
    <div style="margin-top: 10px;">
        <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fsynthomics.local&countColor=%237c3aed" alt="Visitor Count" />
    </div>
</div>
""", unsafe_allow_html=True)
