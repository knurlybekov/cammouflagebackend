import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color


def extract_color_palette_dict(
    image_path: str,
    n_clusters: int = 5,
    use_bilateral: bool = True,
    max_pixels: int = 400_000,
    random_state: int = 42,
    precision: int = 2,          # number of decimals for percentage keys
    percent_keys_as_str: bool = True  # True -> "42.13%", False -> 42.13 (float)
):
    """
    Returns:
        palette_dict: dict mapping percentage -> [R, G, B] (or list of [R,G,B] if tie on rounded %)
        dominant_rgb: (k, 3) uint8 array of cluster RGB colors, sorted by % desc
        dominant_pct: (k,) float array of percentages (0-100), aligned with dominant_rgb
    """
    # ---- Load (BGR) and convert to RGB ----
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image not found or invalid: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---- Optional smoothing ----
    if use_bilateral:
        rgb = cv2.bilateralFilter(rgb, d=15, sigmaColor=75, sigmaSpace=75)

    # ---- Optional downscale for speed ----
    H, W = rgb.shape[:2]
    total_px = H * W
    if total_px > max_pixels:
        scale = (max_pixels / total_px) ** 0.5
        new_size = (max(1, int(W * scale)), max(1, int(H * scale)))
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        H, W = rgb.shape[:2]

    # ---- Lab conversion (skimage expects [0,1] floats) ----
    rgb_norm = rgb.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb_norm)

    # ---- Flatten ----
    lab_flat = lab.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)

    # ---- K-Means in Lab space ----
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(lab_flat)

    # ---- Mean RGB per cluster ----
    dominant_rgb = np.zeros((n_clusters, 3), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for k in range(n_clusters):
        mask = (labels == k)
        counts[k] = mask.sum()
        dominant_rgb[k] = rgb_flat[mask].mean(axis=0)

    # ---- Sort by coverage ----
    dominant_pct = (counts / counts.sum()) * 100.0
    order = np.argsort(-dominant_pct)
    dominant_rgb = np.clip(dominant_rgb[order], 0, 255).astype(np.uint8)
    dominant_pct = dominant_pct[order]

    # ---- Build {percent -> RGB} dictionary ----
    palette_dict = {}
    for pct, rgb_val in zip(dominant_pct, dominant_rgb):
        key = f"{pct:.{precision}f}%" if percent_keys_as_str else round(float(pct), precision)
        # If two clusters round to the same percentage, store both colors in a list
        if key in palette_dict:
            # ensure list of colors
            if isinstance(palette_dict[key][0], int):
                palette_dict[key] = [palette_dict[key]]
            palette_dict[key].append(rgb_val.tolist())
        else:
            palette_dict[key] = rgb_val.tolist()

    return palette_dict, dominant_rgb, dominant_pct


# ===== Example usage =====
image_path = "../media/images/IMG_3783.jpeg"
palette_dict, dominant_rgb, dominant_pct = extract_color_palette_dict(
    image_path,
    n_clusters=5,
    use_bilateral=True,
    max_pixels=400_000,
    precision=2,
    percent_keys_as_str=True
)

# Your reusable dict:
# e.g. {'42.13%': [123, 145, 98], '27.31%': [34, 56, 20], ...}
print(palette_dict)

# If you also want arrays:
# print(dominant_rgb)  # (k,3) uint8
# print(dominant_pct)  # (k,) float, 0-100

"""luminance stuff to get the texture"""

from pathlib import Path
from PIL import Image
import numpy as np

# ---------- Utils ----------
def equalize_histogram(gray_img: Image.Image) -> Image.Image:
    """Robust histogram equalization for 8-bit grayscale."""
    arr = np.asarray(gray_img)
    if arr.ndim != 2:
        raise ValueError("equalize_histogram expects a grayscale (L) image")
    hist = np.bincount(arr.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min, cdf_max = cdf_masked.min(), cdf_masked.max()
    lut = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    lut = np.ma.filled(lut, 0).astype(np.uint8)
    out = lut[arr]
    return Image.fromarray(out, mode="L")

def luminance_channel(img: Image.Image) -> Image.Image:
    """Return luminance (Y) channel preserving color space correctness."""
    return img.convert("L")  # (PIL uses Rec.601; good enough for most cases)

def equalize_luminance_preserve_color(img: Image.Image) -> Image.Image:
    """Equalize luminance while keeping color (YCbCr)."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_eq = equalize_histogram(y)
    return Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")

def segment_boxes(width, height, rows=3, cols=3):
    """Compute non-overlapping boxes that cover the whole image."""
    xs = np.linspace(0, width, cols + 1, dtype=int)
    ys = np.linspace(0, height, rows + 1, dtype=int)
    boxes = []
    for i in range(rows):
        for j in range(cols):
            boxes.append((xs[j], ys[i], xs[j+1], ys[i+1]))
    return boxes

def segment_image(img: Image.Image, rows=3, cols=3):
    """Return list of (tile_img, box)."""
    boxes = segment_boxes(*img.size, rows=rows, cols=cols)
    tiles = [(img.crop(b), b) for b in boxes]
    return tiles

def stitch_tiles(tiles, size):
    """Paste tiles back into a single image (expects matching sizes/boxes)."""
    out = Image.new(tiles[0][0].mode, size)
    for tile_img, box in tiles:
        out.paste(tile_img, box)
    return out

# ---------- Pipelines ----------
def full_image_luminance_equalized(img: Image.Image) -> Image.Image:
    """Equalize luminance of the whole image and return grayscale."""
    L = luminance_channel(img)
    return equalize_histogram(L)

def per_tile_equalized_stitched(img: Image.Image, rows=3, cols=3) -> Image.Image:
    """Equalize luminance per tile, then stitch back (coarse local contrast)."""
    tiles = segment_image(img, rows, cols)
    processed = []
    for tile_img, box in tiles:
        L = luminance_channel(tile_img)
        L_eq = equalize_histogram(L)
        processed.append((L_eq, box))  # output grayscale; swap to color if desired
    return stitch_tiles(processed, img.size)

def block_luminance_map(img: Image.Image, rows=3, cols=3) -> Image.Image:
    """Create a coarse 3×3 luminance map (each block filled with its mean)."""
    tiles = segment_image(img, rows, cols)
    out = Image.new("L", img.size)
    for tile_img, box in tiles:
        L = luminance_channel(tile_img)
        mean_val = int(np.asarray(L).mean())
        w = box[2] - box[0]
        h = box[3] - box[1]
        block = Image.new("L", (w, h), color=mean_val)
        out.paste(block, box)
    return out


def average_texture_from_tiles(img: Image.Image, rows=3, cols=3) -> Image.Image:
    tiles = [t[0] for t in segment_image(img, rows, cols)]
    # resize all to the smallest tile size (they should already match, but safe):
    w, h = min(t.size[0] for t in tiles), min(t.size[1] for t in tiles)
    arrays = [np.asarray(t.convert("L").resize((w, h))) for t in tiles]
    avg = np.mean(arrays, axis=0).astype(np.uint8)
    return equalize_histogram(Image.fromarray(avg, mode="L"))

img = Image.open(image_path).convert("RGB")
some = average_texture_from_tiles(img, rows=3, cols=3)
some.save(image_path+'_luminacetexture.jpg')

