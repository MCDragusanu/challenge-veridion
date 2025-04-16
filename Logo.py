import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import pandas as pd
def get_bgr_matrix_from_url(image_url, target_size=(224, 224)):
    try:
        # 1. Fetch image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # 2. Load with PIL, convert to RGB
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # 3. Center crop to square
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        # 4. Resize to fixed shape (e.g., 224x224)
        img = img.resize(target_size, Image.LANCZOS)

        # 5. Convert to NumPy array and BGR format (for OpenCV-style processing)
        rgb_array = np.array(img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array  # Shape: (224, 224, 3)

    except Exception as e:
        print(f"[Error] Failed to process image from {image_url}: {e}")
        return None

def extract_features_from_logo(logo_bgr_matrix):
    if logo_bgr_matrix is None:
        return pd.DataFrame([{
            "mean_b": None, "mean_g": None, "mean_r": None,
            "std_b": None, "std_g": None, "std_r": None,
            "edge_density": None
        }])

    try:
        # Compute color means and stds
        mean_b = np.mean(logo_bgr_matrix[:, :, 0])
        mean_g = np.mean(logo_bgr_matrix[:, :, 1])
        mean_r = np.mean(logo_bgr_matrix[:, :, 2])

        std_b = np.std(logo_bgr_matrix[:, :, 0])
        std_g = np.std(logo_bgr_matrix[:, :, 1])
        std_r = np.std(logo_bgr_matrix[:, :, 2])

        # Convert to grayscale
        gray = cv2.cvtColor(logo_bgr_matrix, cv2.COLOR_BGR2GRAY)

        # Edge detection (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        # Package into DataFrame
        return pd.DataFrame([{
            "mean_b": mean_b, "mean_g": mean_g, "mean_r": mean_r,
            "std_b": std_b, "std_g": std_g, "std_r": std_r,
            "edge_density": edge_density
        }])

    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return pd.DataFrame([{
            "mean_b": None, "mean_g": None, "mean_r": None,
            "std_b": None, "std_g": None, "std_r": None,
            "edge_density": None
        }])
