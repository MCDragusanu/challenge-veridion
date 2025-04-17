import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
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
        tree_matrix = np.array(feature_list)
        print(f"[✓] Tree_{tree_number}: {tree_matrix.shape[0]} logos loaded.")
        return tree_matrix
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

def compute_similarity_matrix(trees):
    n = len(trees)
    similarity_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing similarity"):
        # Set diagonal to 0 (identical to itself)
        similarity_matrix[i][i] = 0
        
        for j in range(i + 1, n):
            tree_i = trees[i]
            tree_j = trees[j]

            if tree_i is None or tree_j is None:
                similarity_matrix[i][j] = similarity_matrix[j][i] = np.inf
                continue

            # Use minimum distance between any pair of logos as the tree similarity
            summed_distances = 0
            element_count = 0
            for logo_i in tree_i:
                for logo_j in tree_j:
                    # Ensure vectors are same length
                    if len(logo_i) != len(logo_j):
                        max_len = max(len(logo_i), len(logo_j))
                        padded_i = np.zeros(max_len)
                        padded_j = np.zeros(max_len)
                        padded_i[:len(logo_i)] = logo_i
                        padded_j[:len(logo_j)] = logo_j
                        distance = np.linalg.norm(padded_i - padded_j)
                    else:
                        distance = np.linalg.norm(logo_i - logo_j)
                    element_count = element_count + 1
                    summed_distances = summed_distances + distance

            average =  summed_distances / element_count if element_count > 0  else  np.inf           
            # Store the minimum distance as our similarity measure
            similarity_matrix[i][j] = similarity_matrix[j][i] = average

    return similarity_matrix

def save_cluster(cluster_id, cluster_trees, cluster_dir='dataset/clusters'):
    try:
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)  # Create intermediate directories if needed

        cluster_file = os.path.join(cluster_dir, f'cluster_{cluster_id}.xml')
        with open(cluster_file, 'w', encoding='utf-8') as f:
            f.write(f'<Cluster id="{cluster_id}">')
            for tree in cluster_trees:
                f.write(f'\n\t<Tree>{tree}</Tree>')
            f.write('\n</Cluster>')
        
        print(f"[✓] Cluster {cluster_id} saved to {cluster_file}")

    except Exception as e:
        print(f"[✗] Failed to save cluster {cluster_id}: {e}")

similarity_matrix = compute_similarity_matrix(trees)

# Replace any inf values with the maximum finite value
max_finite = np.max(similarity_matrix[np.isfinite(similarity_matrix)])
similarity_matrix[~np.isfinite(similarity_matrix)] = max_finite * 1.5

# Normalize the similarity matrix to [0, 1] range
# Create a copy to preserve the original for reference if needed
raw_similarity_matrix = similarity_matrix.copy()

# Use MinMaxScaler to normalize the matrix values between 0 and 1
# First, we need to reshape the matrix to a 1D array
matrix_shape = similarity_matrix.shape
flattened_matrix = similarity_matrix.flatten()

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_flat = scaler.fit_transform(flattened_matrix.reshape(-1, 1))

# Reshape back to original matrix shape
similarity_matrix = normalized_flat.reshape(matrix_shape)

# Plot normalized distance matrix
plt.figure(figsize=(12, 10))
plt.imshow(similarity_matrix, cmap='viridis_r', interpolation='nearest')
plt.colorbar(label='Normalized Distance (0-1, lower means more similar)')
plt.title('Normalized Logo Tree Distance Matrix')
plt.xlabel('Tree Index')
plt.ylabel('Tree Index')
plt.savefig('logo_distance_matrix_normalized.png')
plt.show()

# Apply hierarchical clustering using the normalized matrix
# Convert distance matrix to condensed form for linkage
condensed_dist = squareform(similarity_matrix)
Z = linkage(condensed_dist, method='average')  # Use average linkage

distance_threshold = 0.1  
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

if os.path.exists(cluster_dir) :
    files = os.listdir(cluster_dir)
    for file in files:
        os.remove(os.path.join(cluster_dir,file))

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
    for cid, trees in mergeable_clusters.items():
        f.write(f"Cluster {cid}: Trees {sorted(trees)} can be merged\n")
        save_cluster(cid , trees)
        used_cluster_ids.append(cid)
        already_clustered_trees.append(trees)

#now create the clusters with 1 members

last_cluster_id = 0

# Flatten the list of lists
flattened_clustered_trees = set(chain.from_iterable(already_clustered_trees))

# Find trees not already assigned to a cluster
not_mapped_trees = [i for i in range(1, original_tree_count + 1) if i not in flattened_clustered_trees]


for tree_id in not_mapped_trees:
    # Ensure unique cluster ID
    while last_cluster_id in used_cluster_ids:
        last_cluster_id += 1

    save_cluster(last_cluster_id, [tree_id])
    used_cluster_ids.append(last_cluster_id)
