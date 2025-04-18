import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform, cosine
from itertools import chain

logo_directory = 'dataset/logos'

def build_tree_matrix(tree_number):
    path = os.path.join(logo_directory, f'tree_{tree_number}')
    files = os.listdir(path)
    feature_list = []
    
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            try:
                features_df = pd.read_csv(file_path)
                features = features_df.values.flatten()
                feature_list.append(features)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if feature_list:
        # Convert to a list instead of directly to numpy array
        print(f"[✓] Tree_{tree_number}: {len(feature_list)} logos loaded.")
        return feature_list
    else:
        print(f"[✗] Tree_{tree_number}: No valid logo feature files.")
        return None

if not os.path.exists(logo_directory):
    raise FileNotFoundError(f"Directory {logo_directory} not found.")

tree_dirs = [d for d in os.listdir(logo_directory) if os.path.isdir(os.path.join(logo_directory, d))]
tree_numbers = sorted([int(d.split('_')[1]) for d in tree_dirs])

trees = []
index_mapping = {}  # tree_number -> index in matrix
reverse_index_mapping = {}  # index in matrix -> tree_number
current_index = 0

for tree_number in tree_numbers:
    features = build_tree_matrix(tree_number)
    if features is not None:
        index_mapping[tree_number] = current_index
        reverse_index_mapping[current_index] = tree_number
        trees.append(features)
        current_index += 1

def compute_improved_similarity_matrix(trees):
    n = len(trees)
    similarity_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing similarity"):
        for j in range(i + 1, n):
            tree_i = trees[i]
            tree_j = trees[j]

            if tree_i is None or tree_j is None or len(tree_i) == 0 or len(tree_j) == 0:
                similarity_matrix[i][j] = similarity_matrix[j][i] = 1.0  # Max distance
                continue

            # Compute pairwise cosine similarities between logos
            # This is more robust than Euclidean distance for different sized feature vectors
            similarities = []
            
            for logo_i in tree_i:
                for logo_j in tree_j:
                    # Handle potential different feature sizes
                    min_size = min(len(logo_i), len(logo_j))
                    if min_size == 0:
                        continue
                    
                    # Use the common features only
                    logo_i_trimmed = logo_i[:min_size]
                    logo_j_trimmed = logo_j[:min_size]
                    
                    # Skip if either vector is all zeros
                    if not np.any(logo_i_trimmed) or not np.any(logo_j_trimmed):
                        continue
                    
                    # Compute cosine similarity and convert to distance (1 - similarity)
                    # Lower distance means more similar logos
                    sim = 1.0 - cosine(logo_i_trimmed, logo_j_trimmed)
                    similarities.append(sim)
            
            if similarities:
                # Take the maximum similarity (minimum distance) as the representative measure
                # This approach assumes that if at least one logo pair is very similar,
                # the trees are potentially related
                max_similarity = max(similarities)
                # Convert to distance (lower means more similar)
                distance = 1.0 - max_similarity
                similarity_matrix[i][j] = similarity_matrix[j][i] = distance
            else:
                similarity_matrix[i][j] = similarity_matrix[j][i] = 1.0  # Max distance

    return similarity_matrix

similarity_matrix = compute_improved_similarity_matrix(trees)

# Ensure all values are valid (no NaN or inf)
similarity_matrix = np.nan_to_num(similarity_matrix, nan=1.0, posinf=1.0, neginf=1.0)

# Scale similarity matrix to [0, 1] range if needed
# (values should already be between 0 and 1 from cosine distance)
scaler = MinMaxScaler(feature_range=(0, 1))
matrix_shape = similarity_matrix.shape
flattened_matrix = similarity_matrix.flatten().reshape(-1, 1)
normalized_flat = scaler.fit_transform(flattened_matrix)
similarity_matrix = normalized_flat.reshape(matrix_shape)

# Plot normalized distance matrix
plt.figure(figsize=(12, 10))
plt.imshow(similarity_matrix, cmap='viridis_r', interpolation='nearest')
plt.colorbar(label='Normalized Distance (0-1, lower means more similar)')
plt.title('Normalized Logo Tree Distance Matrix')
plt.xlabel('Tree Index')
plt.ylabel('Tree Index')
plt.savefig('logo_distance_matrix_normalized.png')

# Plot dendrogram to help with threshold selection
plt.figure(figsize=(14, 8))
condensed_dist = squareform(similarity_matrix)
Z = linkage(condensed_dist, method='ward')  # Changed to ward linkage for better clusters
dendrogram(Z, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Tree Index')
plt.ylabel('Distance')
plt.axhline(y=0.15, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.savefig('logo_clustering_dendrogram.png')
plt.close()  # Close plots to save memory

# Apply hierarchical clustering using the normalized matrix
# Try ward linkage which often produces more balanced clusters
Z = linkage(condensed_dist, method='ward')
distance_threshold = 0.15  # Adjusted threshold based on dendrogram
clusters = fcluster(Z, distance_threshold, criterion='distance')

# Map clusters back to original tree numbers
cluster_mapping = defaultdict(list)
for i, cluster_id in enumerate(clusters):
    original_tree = reverse_index_mapping[i]
    cluster_mapping[cluster_id].append(original_tree)

# Filter to show only clusters with more than one tree
mergeable_clusters = {k: v for k, v in cluster_mapping.items() if len(v) > 1}

# Final stats
original_tree_count = len(tree_numbers)

# Each cluster with size > 1 contributes (size-1) to the reduction
reduction = sum(len(cluster) - 1 for cluster in mergeable_clusters.values())
merged_tree_count = original_tree_count - reduction
reduction_percentage = (reduction / original_tree_count) * 100 if original_tree_count > 0 else 0

print(f"\nOriginal number of trees: {original_tree_count}")
print(f"Distance threshold (normalized): {distance_threshold}")
print(f"Number of trees after merging: {merged_tree_count}")
print(f"Reduction: {reduction} trees ({reduction_percentage:.2f}%)")
print(f"Number of clusters with multiple trees: {len(mergeable_clusters)}")

cluster_dir = 'dataset/clusters'

# Create clusters directory if it doesn't exist
if not os.path.exists(cluster_dir):
    os.makedirs(cluster_dir)
else:
    # Clean existing files
    files = os.listdir(cluster_dir)
    for file in files:
        os.remove(os.path.join(cluster_dir, file))

def save_cluster(cluster_id, cluster_trees, cluster_dir='dataset/clusters'):
    try:
        cluster_file = os.path.join(cluster_dir, f'cluster_{cluster_id}.xml')
        with open(cluster_file, 'w', encoding='utf-8') as f:
            f.write(f'<Cluster id="{cluster_id}">')
            for tree in sorted(cluster_trees):  # Sort trees for consistent output
                f.write(f'\n\t<Tree>{tree}</Tree>')
            f.write('\n</Cluster>')
        
        print(f"[✓] Cluster {cluster_id} saved to {cluster_file}")

    except Exception as e:
        print(f"[✗] Failed to save cluster {cluster_id}: {e}")

used_cluster_ids = []
already_clustered_trees = []

# Save clustering results
with open('tree_clustering_results.txt', 'w') as f:
    f.write("TREE CLUSTERING RESULTS\n")
    f.write("======================\n\n")
    f.write(f"Original number of trees: {original_tree_count}\n")
    f.write(f"Distance threshold (normalized): {distance_threshold}\n")
    f.write(f"Number of trees after merging: {merged_tree_count}\n")
    f.write(f"Reduction: {reduction} trees ({reduction_percentage:.2f}%)\n\n")
    f.write("Mergeable Tree Clusters:\n")
    
    # Save clusters with multiple trees first
    for cid, trees in mergeable_clusters.items():
        f.write(f"Cluster {cid}: Trees {sorted(trees)} can be merged\n")
        save_cluster(cid, trees)
        used_cluster_ids.append(cid)
        already_clustered_trees.extend(trees)

# Now create the clusters with 1 member
last_cluster_id = max(used_cluster_ids) if used_cluster_ids else 0
# Find trees not already assigned to a cluster
not_mapped_trees = [i for i in tree_numbers if i not in already_clustered_trees]

for tree_id in not_mapped_trees:
    # Ensure unique cluster ID
    last_cluster_id += 1
    save_cluster(last_cluster_id, [tree_id])
    used_cluster_ids.append(last_cluster_id)

print(f"\nAll {len(used_cluster_ids)} clusters saved to {cluster_dir}")
print(f"Single-tree clusters: {len(not_mapped_trees)}")
print(f"Multi-tree clusters: {len(mergeable_clusters)}")