"""
Art Module - Generative visualization from DNA sequences.

Styles:
1. Heatmap + DNA Walk (scientific default)
2. Entropy Chaos (organic, complex)
3. Codon Mosaic (geometric, bold)
"""

import math
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

# Try to import visualization libraries
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# COLOR PALETTES
# =============================================================================

# Base colors for nucleotides (scientific palette)
NUCLEOTIDE_COLORS = {
    'A': (76, 175, 80),    # Green (Adenine)
    'T': (244, 67, 54),    # Red (Thymine)
    'G': (33, 150, 243),   # Blue (Guanine)
    'C': (255, 193, 7),    # Yellow/Gold (Cytosine)
    'N': (158, 158, 158),  # Gray (Unknown)
}

# Codon color palette (64 colors, deterministic)
def _generate_codon_palette() -> Dict[str, Tuple[int, int, int]]:
    """Generate deterministic color for each codon."""
    bases = 'ATGC'
    codons = [a + b + c for a in bases for b in bases for c in bases]
    
    palette = {}
    for i, codon in enumerate(codons):
        # Use hash for deterministic but varied colors
        h = hashlib.md5(codon.encode()).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        # Boost saturation
        max_val = max(r, g, b)
        if max_val < 200:
            factor = 200 / max(max_val, 1)
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
        palette[codon] = (r, g, b)
    
    return palette

CODON_PALETTE = _generate_codon_palette()


# =============================================================================
# DNA WALK
# =============================================================================

def compute_dna_walk(seq: str) -> Tuple[List[int], List[int]]:
    """
    Compute 2D DNA walk coordinates.
    
    Mapping:
        A → Up (+y)
        T → Down (-y)
        G → Right (+x)
        C → Left (-x)
    
    Args:
        seq: DNA sequence
        
    Returns:
        (x_coords, y_coords) lists
    """
    x, y = 0, 0
    xs, ys = [0], [0]
    
    for nuc in seq.upper():
        if nuc == 'A':
            y += 1
        elif nuc == 'T':
            y -= 1
        elif nuc == 'G':
            x += 1
        elif nuc == 'C':
            x -= 1
        # N doesn't move
        
        xs.append(x)
        ys.append(y)
    
    return xs, ys


def compute_cumulative_gc_walk(seq: str) -> Tuple[List[int], List[float]]:
    """
    Compute cumulative GC skew walk.
    
    Args:
        seq: DNA sequence
        
    Returns:
        (positions, gc_skew) for plotting
    """
    positions = list(range(len(seq) + 1))
    skew = [0.0]
    
    for nuc in seq.upper():
        if nuc == 'G':
            skew.append(skew[-1] + 1)
        elif nuc == 'C':
            skew.append(skew[-1] - 1)
        else:
            skew.append(skew[-1])
    
    return positions, skew


# =============================================================================
# K-MER HEATMAP
# =============================================================================

def compute_kmer_matrix(seq: str, k: int = 4) -> np.ndarray:
    """
    Compute k-mer frequency matrix for heatmap.
    
    Organizes k-mers as a 2D grid based on prefix/suffix.
    
    Args:
        seq: DNA sequence
        k: k-mer length (must be even, default 4)
        
    Returns:
        2D numpy array of frequencies
    """
    if k % 2 != 0:
        k = k + 1  # Make even
    
    half_k = k // 2
    bases = 'ATGC'
    
    # Generate all half-k-mers for rows and columns
    half_kmers = []
    
    def gen_kmers(prefix, n):
        if n == 0:
            half_kmers.append(prefix)
            return
        for b in bases:
            gen_kmers(prefix + b, n - 1)
    
    gen_kmers('', half_k)
    
    # Count full k-mers
    kmer_counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k].upper()
        if all(c in 'ATGC' for c in kmer):
            kmer_counts[kmer] += 1
    
    # Build matrix
    size = len(half_kmers)
    matrix = np.zeros((size, size), dtype=float)
    
    kmer_to_idx = {km: i for i, km in enumerate(half_kmers)}
    
    for kmer, count in kmer_counts.items():
        prefix = kmer[:half_k]
        suffix = kmer[half_k:]
        if prefix in kmer_to_idx and suffix in kmer_to_idx:
            matrix[kmer_to_idx[prefix], kmer_to_idx[suffix]] = count
    
    # Log scale for better visualization
    matrix = np.log1p(matrix)
    
    return matrix


# =============================================================================
# ENTROPY VISUALIZATION
# =============================================================================

def compute_local_entropy_grid(seq: str, window: int = 50, 
                                grid_width: int = 100) -> np.ndarray:
    """
    Compute local entropy as a 2D grid for visualization.
    
    Args:
        seq: DNA sequence
        window: Entropy calculation window
        grid_width: Width of output grid
        
    Returns:
        2D numpy array of entropy values
    """
    from . import stats
    
    # Calculate local entropy
    entropies = []
    for i in range(0, len(seq), window // 2):
        chunk = seq[i:i + window]
        if len(chunk) >= window // 2:
            entropies.append(stats.calculate_entropy(chunk))
    
    if not entropies:
        return np.zeros((10, 10))
    
    # Reshape into grid
    n = len(entropies)
    height = max(1, n // grid_width + 1)
    
    # Pad to fill grid
    padded = entropies + [0] * (height * grid_width - n)
    grid = np.array(padded[:height * grid_width]).reshape(height, grid_width)
    
    return grid


# =============================================================================
# STYLE 1: HEATMAP + DNA WALK
# =============================================================================

def generate_heatmap_walk_art(seq: str, 
                               width: int = 800, 
                               height: int = 800,
                               colormap: str = 'viridis') -> Image.Image:
    """
    Generate Heatmap + DNA Walk artwork.
    
    Args:
        seq: DNA sequence
        width: Image width
        height: Image height
        colormap: Matplotlib colormap name
        
    Returns:
        PIL Image
    """
    if not HAS_PIL or not HAS_MATPLOTLIB:
        # Fallback to simple image
        return _generate_simple_art(seq, width, height)
    
    # Create figure
    fig = Figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Generate k-mer heatmap
    kmer_matrix = compute_kmer_matrix(seq, k=4)
    
    # Plot heatmap
    im = ax.imshow(kmer_matrix, cmap=colormap, aspect='auto', 
                   extent=[0, width, 0, height], alpha=0.7)
    
    # Compute DNA walk
    xs, ys = compute_dna_walk(seq)
    
    # Normalize walk to image coordinates
    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        x_range = max(x_max - x_min, 1)
        y_range = max(y_max - y_min, 1)
        
        xs_norm = [(x - x_min) / x_range * (width * 0.8) + width * 0.1 for x in xs]
        ys_norm = [(y - y_min) / y_range * (height * 0.8) + height * 0.1 for y in ys]
        
        # Color walk by position
        points = np.array([xs_norm, ys_norm]).T.reshape(-1, 1, 2)
        if len(points) > 1:
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            from matplotlib.collections import LineCollection
            colors = plt.cm.plasma(np.linspace(0, 1, len(segments)))
            lc = LineCollection(segments, colors=colors, linewidth=1.5, alpha=0.8)
            ax.add_collection(lc)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    # Convert to PIL Image
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)
    
    plt.close(fig)
    
    return img.convert('RGB')


# =============================================================================
# STYLE 2: ENTROPY CHAOS
# =============================================================================

def generate_chaos_art(seq: str,
                        width: int = 800,
                        height: int = 800) -> Image.Image:
    """
    Generate entropy-based chaos visualization.
    
    Uses chaos game representation with entropy-driven coloring.
    
    Args:
        seq: DNA sequence
        width: Image width
        height: Image height
        
    Returns:
        PIL Image
    """
    if not HAS_PIL:
        raise ImportError("PIL required for art generation")
    
    # Create image
    img = Image.new('RGB', (width, height), (10, 10, 30))
    pixels = img.load()
    
    # Chaos game corners (tetrahedral projection to 2D)
    corners = {
        'A': (width // 2, 50),
        'T': (width // 2, height - 50),
        'G': (width - 50, height // 2),
        'C': (50, height // 2),
    }
    
    # Initialize position at center
    x, y = width // 2, height // 2
    
    # Chaos game iteration
    from . import stats
    
    # Precompute local entropies for coloring
    window = 20
    local_entropies = []
    for i in range(0, len(seq), window):
        chunk = seq[i:i+window]
        local_entropies.append(stats.calculate_entropy(chunk))
    
    max_entropy = max(local_entropies) if local_entropies else 2.0
    
    # Run chaos game
    for i, nuc in enumerate(seq.upper()):
        if nuc not in 'ATGC':
            continue
        
        # Move halfway to corner
        corner = corners[nuc]
        x = (x + corner[0]) // 2
        y = (y + corner[1]) // 2
        
        # Color based on local entropy
        entropy_idx = min(i // window, len(local_entropies) - 1)
        entropy = local_entropies[entropy_idx] if local_entropies else 1.0
        entropy_norm = entropy / max(max_entropy, 0.01)
        
        # Entropy-based color (low=blue, high=red/orange)
        r = int(50 + entropy_norm * 200)
        g = int(100 + (1 - entropy_norm) * 100)
        b = int(200 - entropy_norm * 150)
        
        # Plot point with slight randomness for organic feel
        px = max(0, min(width - 1, x))
        py = max(0, min(height - 1, y))
        
        # Additive blending
        old = pixels[px, py]
        pixels[px, py] = (
            min(255, old[0] + r // 10),
            min(255, old[1] + g // 10),
            min(255, old[2] + b // 10)
        )
    
    # Apply slight blur for smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


# =============================================================================
# STYLE 3: CODON MOSAIC
# =============================================================================

def generate_codon_mosaic(seq: str,
                           width: int = 800,
                           height: int = 800,
                           border: bool = True) -> Image.Image:
    """
    Generate codon-colored mosaic grid.
    
    Args:
        seq: DNA sequence
        width: Image width
        height: Image height
        border: Draw borders between tiles
        
    Returns:
        PIL Image
    """
    if not HAS_PIL:
        raise ImportError("PIL required for art generation")
    
    # Extract codons
    seq = seq.upper()
    codons = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if all(c in 'ATGC' for c in codon):
            codons.append(codon)
    
    if not codons:
        return Image.new('RGB', (width, height), (100, 100, 100))
    
    # Calculate grid dimensions
    n_codons = len(codons)
    cols = int(math.ceil(math.sqrt(n_codons * width / height)))
    rows = int(math.ceil(n_codons / cols))
    
    tile_w = width // cols
    tile_h = height // rows
    
    # Create image
    img = Image.new('RGB', (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Draw tiles
    for i, codon in enumerate(codons):
        row = i // cols
        col = i % cols
        
        x0 = col * tile_w
        y0 = row * tile_h
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        
        color = CODON_PALETTE.get(codon, (128, 128, 128))
        
        if border:
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=color)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=color)
    
    return img


# =============================================================================
# SIMPLE FALLBACK
# =============================================================================

def _generate_simple_art(seq: str, width: int, height: int) -> Image.Image:
    """Generate simple art without matplotlib."""
    if not HAS_PIL:
        raise ImportError("PIL required")
    
    img = Image.new('RGB', (width, height), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    
    # Draw DNA walk
    xs, ys = compute_dna_walk(seq)
    
    if len(xs) < 2:
        return img
    
    # Normalize
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)
    
    points = []
    for x, y in zip(xs, ys):
        px = int((x - x_min) / x_range * (width * 0.8) + width * 0.1)
        py = int((y - y_min) / y_range * (height * 0.8) + height * 0.1)
        points.append((px, py))
    
    # Draw line segments with color gradient
    for i in range(len(points) - 1):
        progress = i / max(len(points) - 1, 1)
        r = int(50 + progress * 200)
        g = int(150 - progress * 100)
        b = int(200 - progress * 150)
        draw.line([points[i], points[i + 1]], fill=(r, g, b), width=2)
    
    return img


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

ART_STYLES = {
    'heatmap_walk': {
        'name': 'Heatmap + DNA Walk',
        'description': 'K-mer frequency heatmap with DNA walk trajectory overlay',
        'function': generate_heatmap_walk_art
    },
    'chaos': {
        'name': 'Entropy Chaos',
        'description': 'Chaos game representation with entropy-driven coloring',
        'function': generate_chaos_art
    },
    'mosaic': {
        'name': 'Codon Mosaic',
        'description': 'Colorful grid where each tile represents a codon',
        'function': generate_codon_mosaic
    }
}


def generate_art(seq: str,
                 style: str = 'heatmap_walk',
                 width: int = 800,
                 height: int = 800,
                 **kwargs) -> Image.Image:
    """
    Generate artwork from DNA sequence.
    
    Args:
        seq: DNA sequence
        style: Art style ('heatmap_walk', 'chaos', 'mosaic')
        width: Image width
        height: Image height
        **kwargs: Additional style-specific arguments
        
    Returns:
        PIL Image
    """
    if style not in ART_STYLES:
        style = 'heatmap_walk'
    
    func = ART_STYLES[style]['function']
    return func(seq, width=width, height=height, **kwargs)


def save_art(img: Image.Image, filepath: str, format: str = 'PNG'):
    """Save artwork to file."""
    img.save(filepath, format=format)


def get_style_info() -> List[Dict]:
    """Get information about available art styles."""
    return [
        {
            'id': style_id,
            'name': info['name'],
            'description': info['description']
        }
        for style_id, info in ART_STYLES.items()
    ]


def get_art_mapping_legend() -> str:
    """Get human-readable art mapping legend."""
    return """
## DNA → Art Mapping

### DNA Walk (Spatial Trajectory)
| Nucleotide | Direction |
|------------|-----------|
| A (Adenine) | Up (+Y) |
| T (Thymine) | Down (-Y) |
| G (Guanine) | Right (+X) |
| C (Cytosine) | Left (-X) |

### K-mer Heatmap
- Divides sequence into overlapping 4-mers
- Counts frequency of each pattern
- Maps to color intensity (log scale)

### Entropy Coloring
- Low entropy (repetitive) → Cool colors (blue)
- High entropy (random) → Warm colors (red/orange)

### Codon Mosaic
- Each 3-letter codon gets a unique color
- 64 possible codons = 64 colors
- Colors are deterministic (same codon = same color)
"""
