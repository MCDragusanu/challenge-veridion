"""
Cluster Management System
------------------------
This module provides functionality for clustering and merging similar data points
based on feature vector analysis using K-Means clustering.
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Set, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


CLUSTER_DIRECTORY = 'dataset/clusters'
LOGO_DIRECTORY = 'dataset/logos'


def get_tree_ids_from_cluster(cluster_path: str) -> List[str]:
    """Extract tree IDs from a cluster file.
    
    Args:
        cluster_path: Path to the cluster XML file
        
    Returns:
        List of tree IDs contained in the cluster
    """
    try:
        with open(cluster_path, 'r') as f:
            content = f.read()
            return re.findall(r"<Tree>(.*?)<\/Tree>", content) or []
    except Exception as e:
        logger.error(f"Error reading cluster file {cluster_path}: {e}")
        return []


def load_tree_features(tree_id: str) -> List[Tuple[np.ndarray, str]]:
    """Load feature vectors for a tree ID.
    
    Args:
        tree_id: ID of the tree
        
    Returns:
        List of tuples containing (feature_vector, file_path)
    """
    path = os.path.join(LOGO_DIRECTORY, f'tree_{tree_id}')
    if not os.path.exists(path):
        return []
    
    features = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            try:
                df = pd.read_csv(file_path)
                vec = df.values.flatten()
                features.append((vec, file_path))
            except Exception as e:
                logger.warning(f"Error loading feature file {file_path}: {e}")
                continue
    return features


def load_cluster_features(cluster_id: str) -> List[Tuple[np.ndarray, str]]:
    """Load all feature vectors for a cluster.
    
    Args:
        cluster_id: ID of the cluster
        
    Returns:
        List of tuples containing (feature_vector, file_path)
    """
    cluster_path = os.path.join(CLUSTER_DIRECTORY, f'cluster_{cluster_id}.xml')
    if not os.path.exists(cluster_path):
        logger.warning(f"Cluster file not found: {cluster_path}")
        return []
    
    tree_ids = get_tree_ids_from_cluster(cluster_path)
    all_features = []
    for tid in tree_ids:
        all_features.extend(load_tree_features(tid))
    return all_features


def find_central_member(feature_data: List[Tuple[np.ndarray, str]]) -> Optional[str]:
    """Find the most central member (closest to the mean) from feature data.
    
    Args:
        feature_data: List of tuples (feature_vector, file_path)
        
    Returns:
        Path of the most representative member or None if no valid data
    """
    if not feature_data:
        return None

    # Group feature vectors by length
    by_length = {}
    for vec, path in feature_data:
        by_length.setdefault(len(vec), []).append((vec, path))

    if not by_length:
        return None

    # Use the most common feature length
    common_len = max(by_length, key=lambda k: len(by_length[k]))
    data = by_length[common_len]
    vectors = [x[0] for x in data]
    paths = [x[1] for x in data]

    # Calculate the centroid (mean vector)
    centroid = np.mean(vectors, axis=0)

    # Compute Euclidean distances (normed)
    dists = [np.linalg.norm(v - centroid) for v in vectors]

    # Find the index of the closest vector (minimum distance)
    idx = np.argmin(dists)

    return paths[idx]


def pick_cluster_representative(cluster_id: str) -> Optional[str]:
    """Select a representative for a cluster.
    
    Args:
        cluster_id: ID of the cluster
        
    Returns:
        Path to the representative feature file or None
    """
    logger.info(f'Picking representative for cluster {cluster_id}')
    features = load_cluster_features(cluster_id)
    if not features:
        logger.warning(f"No features found for cluster {cluster_id}")
        return None
    
    return find_central_member(features)


def group_representatives_by_vector_length(representatives: Dict[str, str]) -> Dict[int, Dict[str, np.ndarray]]:
    """Group representatives by their feature vector length.
    
    Args:
        representatives: Dictionary mapping cluster IDs to representative paths
        
    Returns:
        Dictionary mapping vector lengths to {cluster_id: vector} dictionaries
    """
    rep_by_length = {}
    for cid, path in representatives.items():
        try:
            vec = pd.read_csv(path).values.flatten()
            rep_by_length.setdefault(len(vec), {})[cid] = vec
        except Exception as e:
            logger.warning(f"Error loading representative {path}: {e}")
            continue
    return rep_by_length


def apply_kmeans_clustering(vectors: np.ndarray, cluster_ids: List[str], k: int) -> Dict[int, List[str]]:
    """Apply K-Means clustering to vectors.
    
    Args:
        vectors: Array of feature vectors
        cluster_ids: List of cluster IDs corresponding to vectors
        k: Number of clusters to form
        
    Returns:
        Dictionary mapping cluster labels to lists of cluster IDs
    """
    if len(vectors) <= 1 or k >= len(vectors):
        # Don't cluster if too few vectors or k is too large
        return {0: cluster_ids}
    
    logger.info(f"Applying KMeans with k={k} on {len(vectors)} vectors")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(vectors)
    
    # Group by cluster labels
    label_to_group = {}
    for idx, cid in enumerate(cluster_ids):
        label = kmeans.labels_[idx]
        if label not in label_to_group:
            label_to_group[label] = []
        label_to_group[label].append(cid)
    
    return label_to_group


def calculate_similarity_scores(group: List[str], vectors: Dict[str, np.ndarray]) -> List[Tuple[str, str, float]]:
    """Calculate similarity scores between all pairs in a group.
    
    Args:
        group: List of cluster IDs
        vectors: Dictionary mapping cluster IDs to feature vectors
        
    Returns:
        List of tuples (cluster_id1, cluster_id2, similarity_score)
    """
    pairs = []
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            cid1, cid2 = group[i], group[j]
            vec1, vec2 = vectors[cid1], vectors[cid2]
            # Calculate cosine similarity (1 - cosine distance)
            try:
                similarity = 1 - cosine(vec1, vec2)
                # Handle NaN values that can occur with zero vectors
                if np.isnan(similarity):
                    similarity = 0
            except Exception:
                similarity = 0
            pairs.append((cid1, cid2, similarity))
    return pairs


def find_mergeable_clusters(representatives: Dict[str, str], target_clusters: Optional[int] = None) -> Tuple[List[List[str]], List[Tuple[str, str, float]]]:
    """Find clusters that can be merged based on representative similarity.
    
    Args:
        representatives: Dictionary mapping cluster IDs to representative paths
        target_clusters: Target number of clusters to aim for
        
    Returns:
        Tuple of (list of mergeable groups, list of similarity scores)
    """
    total_clusters = len(representatives)
    logger.info(f"Finding mergeable clusters with target {target_clusters} from {total_clusters} clusters")
    
    # Group by vector length
    rep_by_length = group_representatives_by_vector_length(representatives)
    
    groups, pairs = [], []
    
    # Process each group of representatives with the same feature vector length
    for vec_len, vecs in rep_by_length.items():
        cluster_ids = list(vecs.keys())
        vectors = np.array([vecs[cid] for cid in cluster_ids])
        
        # Calculate appropriate number of clusters for this group
        if target_clusters is None or target_clusters >= total_clusters:
            # If no target or target is higher than current count, don't merge
            k = len(cluster_ids)
        else:
            # Scale proportionally to maintain the overall ratio
            reduction_ratio = target_clusters / total_clusters
            k = max(1, int(len(cluster_ids) * reduction_ratio))
        
        # Apply clustering
        label_to_group = apply_kmeans_clustering(vectors, cluster_ids, k)
        
        # Add all groups with more than one member
        for group in label_to_group.values():
            if len(group) > 1:
                groups.append(group)
                # Calculate pairwise similarities
                pairs.extend(calculate_similarity_scores(group, vecs))
    
    # Sort pairs by similarity (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return groups, pairs


def merge_cluster_files(group: List[str]) -> Dict[str, Any]:
    """Merge a group of clusters into the first cluster in the group.
    
    Args:
        group: List of cluster IDs to merge
        
    Returns:
        Dictionary with merge information
    """
    group = sorted(group)
    primary = group[0]
    primary_path = os.path.join(CLUSTER_DIRECTORY, f'cluster_{primary}.xml')
    
    # Collect all tree IDs
    trees = []
    for cid in group:
        cluster_path = os.path.join(CLUSTER_DIRECTORY, f'cluster_{cid}.xml')
        trees += get_tree_ids_from_cluster(cluster_path)
    
    # Remove duplicates and sort
    trees = sorted(set(trees))
    
    # Write merged cluster file
    with open(primary_path, 'w') as f:
        f.write(f'<Cluster id="{primary}">\n')
        for tid in trees:
            f.write(f'\t<Tree>{tid}</Tree>\n')
        f.write('</Cluster>')
    
    # Remove other cluster files
    for cid in group[1:]:
        secondary_path = os.path.join(CLUSTER_DIRECTORY, f'cluster_{cid}.xml')
        if os.path.exists(secondary_path):
            os.remove(secondary_path)
    
    return {
        'primary': primary,
        'merged_with': group[1:],
        'tree_count': len(trees)
    }


def perform_cluster_merging(groups: List[List[str]]) -> Dict[str, Any]:
    """Merge all groups of clusters.
    
    Args:
        groups: List of lists containing mergeable cluster IDs
        
    Returns:
        Dictionary with merge statistics
    """
    logger.info(f"Performing merge operation on {len(groups)} groups")
    merged_clusters = {}
    
    for group in groups:
        merge_info = merge_cluster_files(group)
        merged_clusters[merge_info['primary']] = merge_info
    
    return {
        'merged_count': len(groups),
        'merged_clusters': merged_clusters
    }


def get_all_cluster_ids() -> List[str]:
    """Get all cluster IDs from the cluster directory.
    
    Returns:
        List of cluster IDs
    """
    if not os.path.exists(CLUSTER_DIRECTORY):
        logger.error(f"Cluster directory not found: {CLUSTER_DIRECTORY}")
        return []
    
    cluster_files = [f for f in os.listdir(CLUSTER_DIRECTORY) 
                    if f.startswith('cluster_') and f.endswith('.xml')]
    
    return [f.split(".")[0].split("_")[1] for f in cluster_files]


def process_all_clusters(target_n_clusters: Optional[int] = None, perform_merge: bool = True) -> Dict[str, Any]:
    """Process all clusters, find representatives, and potentially merge them.
    
    Args:
        target_n_clusters: Target number of clusters after merging
        perform_merge: Whether to actually perform the merge operation
        
    Returns:
        Dictionary with process statistics
    """
    # Get all cluster IDs
    cluster_ids = get_all_cluster_ids()
    original_cluster_count = len(cluster_ids)
    
    logger.info(f"Processing {original_cluster_count} original clusters")
    
    # Find representatives for each cluster
    representatives = {}
    for cid in cluster_ids:
        rep_path = pick_cluster_representative(cid)
        if rep_path:
            representatives[cid] = rep_path
    
    logger.info(f"Selected representatives for {len(representatives)} clusters")
    
    if not representatives:
        logger.warning("No cluster representatives found. Skipping merge.")
        return {
            'original_cluster_count': original_cluster_count,
            'representatives': {},
            'mergeable_groups': [],
            'mergeable_pairs': [],
            'new_cluster_count': original_cluster_count,
            'reduction_percent': 0,
            'merge_info': None
        }
    
    # Find mergeable clusters
    groups, pairs = find_mergeable_clusters(representatives, target_n_clusters)
    
    # Calculate statistics
    merged_ids = {cid for group in groups for cid in group[1:]}
    new_cluster_count = original_cluster_count - len(merged_ids)
    
    logger.info(f"Original: {original_cluster_count}, Merged: {len(merged_ids)}, New Total: {new_cluster_count}")
    
    merge_info = None
    if perform_merge and groups:
        merge_info = perform_cluster_merging(groups)
        logger.info(f"Merged {merge_info['merged_count']} groups")
    
    # Calculate reduction percentage
    reduction_percent = 0
    if original_cluster_count > 0:
        reduction_percent = len(merged_ids) / original_cluster_count * 100
    
    return {
        'original_cluster_count': original_cluster_count,
        'representatives': representatives,
        'mergeable_groups': groups,
        'mergeable_pairs': pairs,
        'new_cluster_count': new_cluster_count,
        'reduction_percent': reduction_percent,
        'merge_info': merge_info
    }


if __name__ == "__main__":
   
    target_clusters = 150  
    results = process_all_clusters(target_n_clusters=target_clusters, perform_merge=True)
     
        # Stop iterations if no more merging occurs or target reached
    if results['original_cluster_count'] == results['new_cluster_count']:
            logger.info("No more clusters were merged.")
            
        
    if target_clusters is not None and results['new_cluster_count'] <= target_clusters:
            logger.info(f"Target cluster count {target_clusters} reached.")
            