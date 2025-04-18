import requests
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import cv2
import pandas as pd

def remove_background_pil(image):
    """Attempts to remove a solid color background from a PIL Image and make it transparent."""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    width, height = image.size
    bg_color = image.getpixel((0, 0))  # Assume top-left pixel is representative of background
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    distance_tolerance = 20
    # Iterate through all pixels and make background transparent
    for x in range(width):
        for y in range(height):
            current_color = image.getpixel((x, y))
            # Check if the current pixel is "close enough" to the background color
            if all(abs(current_color[i] - bg_color[i]) < distance_tolerance for i in range(3)):
                draw.point((x, y), (0, 0, 0, 0))  # Make transparent
            else:
                draw.point((x, y), current_color)

    return mask

def get_bgr_matrix_from_url(image_url, target_size=(224, 224)):
    if not image_url or not isinstance(image_url, str):
        print("[Error] Invalid image URL")
        return None

    try:
        # 1. Fetch image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # 2. Load with PIL, convert to RGBA initially for transparency handling
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # --- Background Removal Attempt ---
        img_without_bg = remove_background_pil(img)
        img_rgb = img_without_bg.convert("RGB") # Convert back to RGB for feature extraction

        # 3. Center crop to square
        width, height = img_rgb.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img_cropped = img_rgb.crop((left, top, left + min_dim, top + min_dim))

        # 4. Resize to fixed shape (e.g., 224x224)
        img_resized = img_cropped.resize(target_size, Image.LANCZOS)

        # 5. Convert to NumPy array and BGR format (for OpenCV-style processing)
        rgb_array = np.array(img_resized)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array  # Shape: (224, 224, 3)

    except Exception as e:
        print(f"[Error] Failed to process image from {image_url}: {e}")
        return None

def extract_features_from_logo(logo_bgr_matrix):
    if logo_bgr_matrix is None:
        return None

    try:
        features = {}

        # --- 1. Color Means and Stds (6 features) ---
        for i, color in enumerate(['b', 'g', 'r']):
            channel = logo_bgr_matrix[:, :, i]
            features[f"mean_{color}"] = np.mean(channel)
            features[f"std_{color}"] = np.std(channel)

        # --- 2. Color Histograms (3 bins per channel = 9 features) ---
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([logo_bgr_matrix], [i], None, [3], [0, 256]).flatten()
            hist = hist / np.sum(hist)  # Normalize
            for j in range(3):
                features[f"hist_{color}_{j}"] = hist[j]

        # --- 3. Edge Density (1 feature) ---
        gray = cv2.cvtColor(logo_bgr_matrix, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        features["edge_density"] = np.sum(edges > 0) / edges.size

        # --- 4. Shape: Hu Moments (7 features) ---
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i in range(7):
            features[f"hu_moment_{i}"] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)

        # --- 5. Texture: Mean/Std of pixel differences (13 features) ---
        # Use simple texture features instead of full Haralick for speed
        diffs = []
        for dx, dy in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
            shifted = np.roll(gray, shift=(dy, dx), axis=(0, 1))
            diff = np.abs(gray.astype(int) - shifted.astype(int))
            diffs.append(diff)

        all_diffs = np.stack(diffs, axis=0)
        features["texture_mean"] = np.mean(all_diffs)
        features["texture_std"] = np.std(all_diffs)
        for i, d in enumerate(diffs):
            features[f"texture_dir_{i}_mean"] = np.mean(d)
            features[f"texture_dir_{i}_std"] = np.std(d)

        # Padding to 36 (if you ever change features above)
        while len(features) < 36:
            features[f"padding_{len(features)}"] = 0.0

        return pd.DataFrame([features])

    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return None


