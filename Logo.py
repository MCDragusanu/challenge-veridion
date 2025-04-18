import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
import os


def get_bgr_matrix_from_url(image_url, target_size=(224, 224)):
    if not image_url or not isinstance(image_url, str):
        print("[Error] Invalid image URL")
        return None

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")

        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        img = img.resize(target_size, Image.LANCZOS)

        rgb_array = np.array(img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    except Exception as e:
        print(f"[Error] Failed to process image from {image_url}: {e}")
        return None


def get_bgr_matrix_from_file(image_path, target_size=(224, 224)):
    if not image_path or not isinstance(image_path, str):
        print("[Error] Invalid image path")
        return None

    try:
        img = Image.open(image_path).convert("RGB")

        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        img = img.resize(target_size, Image.LANCZOS)

        rgb_array = np.array(img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    except Exception as e:
        print(f"[Error] Failed to process image from {image_path}: {e}")
        return None


# --- Color Feature Functions ---

def compute_colorfulness(bgr_img):
    R, G, B = bgr_img[:, :, 2], bgr_img[:, :, 1], bgr_img[:, :, 0]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return colorfulness


def extract_color_histogram(bgr_img, bins=16):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    
    # Extract histograms for each channel
    h_hist = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Combine all features
    hist_features = {}
    for i, h in enumerate(h_hist):
        hist_features[f'h_hist_{i}'] = float(h)
    for i, s in enumerate(s_hist):
        hist_features[f's_hist_{i}'] = float(s)
    for i, v in enumerate(v_hist):
        hist_features[f'v_hist_{i}'] = float(v)
    
    return hist_features


def extract_dominant_colors(bgr_img, k=5):
    """Extract dominant colors using K-means clustering"""
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    reshaped = hsv_img.reshape((-1, 3))
    reshaped = np.float32(reshaped)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixels in each cluster
    counts = np.bincount(labels.flatten())
    # Sort clusters by size (descending)
    sorted_indices = np.argsort(counts)[::-1]
    
    dominant_color_features = {}
    for i, idx in enumerate(sorted_indices[:k]):
        center = centers[idx]
        percentage = counts[idx] / len(labels.flatten())
        
        dominant_color_features[f'dom_h_{i}'] = float(center[0])
        dominant_color_features[f'dom_s_{i}'] = float(center[1])
        dominant_color_features[f'dom_v_{i}'] = float(center[2])
        dominant_color_features[f'dom_perc_{i}'] = float(percentage)
    
    return dominant_color_features


def compute_color_moments(bgr_img):
    """Calculate color moments (mean, std, skewness, kurtosis) for each channel"""
    moments = {}
    
    # Loop through each channel
    for i, channel_name in enumerate(['b', 'g', 'r']):
        channel = bgr_img[:, :, i]
        # Calculate moments
        moments[f'mean_{channel_name}'] = float(np.mean(channel))
        moments[f'std_{channel_name}'] = float(np.std(channel))
        moments[f'skew_{channel_name}'] = float(skew(channel.flatten()))
        moments[f'kurt_{channel_name}'] = float(kurtosis(channel.flatten()))
    
    return moments


def compute_color_entropy(bgr_img, bins=32):
    hist = cv2.calcHist([bgr_img], [0, 1, 2], None, [bins]*3, [0, 256]*3)
    hist_norm = hist / np.sum(hist)
    hist_flat = hist_norm.flatten()
    entropy_value = entropy(hist_flat + 1e-7, base=2)
    return entropy_value


# --- Shape Feature Functions ---

def extract_shape_features(bgr_img):
    """Extract shape-based features from the logo image"""
    # Convert to grayscale
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to create binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize shape features
    shape_features = {
        'num_contours': len(contours),
        'total_area': 0,
        'total_perimeter': 0,
        'avg_circularity': 0,
        'avg_aspect_ratio': 0,
        'avg_extent': 0,
        'avg_solidity': 0
    }
    
    # If no contours found, return early with zeros
    if len(contours) == 0:
        return shape_features
    
    # Extract individual contour features for the largest 5 contours
    total_circularity = 0
    total_aspect_ratio = 0
    total_extent = 0
    total_solidity = 0
    
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Limit to max 5 contours
    contours = contours[:min(5, len(contours))]
    
    for i, contour in enumerate(contours):
        # Area
        area = cv2.contourArea(contour)
        shape_features[f'contour_{i+1}_area'] = float(area)
        shape_features['total_area'] += float(area)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        shape_features[f'contour_{i+1}_perimeter'] = float(perimeter)
        shape_features['total_perimeter'] += float(perimeter)
        
        # Circularity: 4*pi*area / (perimeter^2)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        shape_features[f'contour_{i+1}_circularity'] = float(circularity)
        total_circularity += circularity
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        shape_features[f'contour_{i+1}_width'] = float(w)
        shape_features[f'contour_{i+1}_height'] = float(h)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        shape_features[f'contour_{i+1}_aspect_ratio'] = float(aspect_ratio)
        total_aspect_ratio += aspect_ratio
        
        # Extent: contour area / bounding rectangle area
        extent = float(area) / (w * h) if w * h > 0 else 0
        shape_features[f'contour_{i+1}_extent'] = float(extent)
        total_extent += extent
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity: contour area / convex hull area
        solidity = float(area) / hull_area if hull_area > 0 else 0
        shape_features[f'contour_{i+1}_solidity'] = float(solidity)
        total_solidity += solidity
    
    # Calculate averages
    num_contours = len(contours)
    if num_contours > 0:
        shape_features['avg_circularity'] = float(total_circularity / num_contours)
        shape_features['avg_aspect_ratio'] = float(total_aspect_ratio / num_contours)
        shape_features['avg_extent'] = float(total_extent / num_contours)
        shape_features['avg_solidity'] = float(total_solidity / num_contours)
    
    return shape_features


def compute_symmetry_score(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    left = gray[:, :w // 2]
    right = np.fliplr(gray[:, w - w // 2:])
    top = gray[:h // 2, :]
    bottom = np.flipud(gray[h - h // 2:, :])

    left_diff = np.mean(np.abs(left - right))
    top_diff = np.mean(np.abs(top - bottom))

    symmetry_score = 1.0 / (1e-5 + left_diff + top_diff)
    return symmetry_score


def compute_center_of_mass(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    total_intensity = np.sum(gray) + 1e-7

    x_center = np.sum(x * gray) / total_intensity
    y_center = np.sum(y * gray) / total_intensity

    return x_center / w, y_center / h


def compute_hu_moments(bgr_img):
    """Compute Hu moments for shape recognition invariant to position, size and rotation"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    
    # Calculate Hu moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform the Hu moments to make them more manageable
    hu_features = {}
    for i, hu in enumerate(hu_moments):
        if hu != 0:
            hu_features[f'hu_moment_{i}'] = float(-np.sign(hu) * np.log10(abs(hu)))
        else:
            hu_features[f'hu_moment_{i}'] = 0.0
    
    return hu_features


def compute_edge_features(bgr_img):
    """Compute edge-related features"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate edge magnitude and direction
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_edge_magnitude = np.mean(magnitude)
    std_edge_magnitude = np.std(magnitude)
    
    edge_features = {
        'edge_density': float(edge_density),
        'mean_edge_magnitude': float(mean_edge_magnitude),
        'std_edge_magnitude': float(std_edge_magnitude)
    }
    
    return edge_features


# --- Main Feature Extraction ---

def extract_features_from_logo(logo_bgr_matrix):
    if logo_bgr_matrix is None:
        return None

    try:
        # Extract basic color statistics
        color_moments = compute_color_moments(logo_bgr_matrix)
        
        # Extract color histograms
        color_histogram = extract_color_histogram(logo_bgr_matrix)
        
        # Extract dominant colors
        dominant_colors = extract_dominant_colors(logo_bgr_matrix)
        
        # Extract shape features
        shape_features = extract_shape_features(logo_bgr_matrix)
        
        # Extract edge features
        edge_features = compute_edge_features(logo_bgr_matrix)
        
        # Compute Hu moments
        hu_moments = compute_hu_moments(logo_bgr_matrix)
        
        # Compute color entropy
        color_entropy_val = compute_color_entropy(logo_bgr_matrix)
        
        # Extract symmetry and center of mass
        symmetry_score = compute_symmetry_score(logo_bgr_matrix)
        x_center, y_center = compute_center_of_mass(logo_bgr_matrix)
        
        # Calculate colorfulness
        colorfulness = compute_colorfulness(logo_bgr_matrix)
        
        # Combine all features
        features = {
            # Color measures
            "colorfulness": float(colorfulness),
            "color_entropy": float(color_entropy_val),
            "symmetry_score": float(symmetry_score),
            "center_x": float(x_center),
            "center_y": float(y_center),
        }
        
        # Add all extracted features
        features.update(color_moments)
        features.update(color_histogram)
        features.update(dominant_colors)
        features.update(shape_features)
        features.update(edge_features)
        features.update(hu_moments)

        return pd.DataFrame([features])

    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return None


