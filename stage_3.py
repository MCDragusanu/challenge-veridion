import os
import re
import numpy as np
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from itertools import chain
from collections import defaultdict

# Load keywords from tree file
def load_keywords_from_tree(tree_id, tree_directory='dataset/trees'):
    path = os.path.join(tree_directory, f'tree_{tree_id}.xml')
    if not os.path.exists(path):
        print(f"[x] Couldn't find {path} file")
        return []

    try:
        with open(path, 'r') as f:
            content = f.read()

        keywords = re.findall(r"<Keyword>(.*?)<\/Keyword>", content, flags=re.DOTALL)
        clean_keywords = [
            re.sub(r'[^a-zA-Z0-9\-]', '', kw.strip().lower()) for kw in keywords
        ]
        return [kw for kw in clean_keywords if kw]

    except Exception as e:
        print(f"[X] Failed to load tree {tree_id}: {e}")
        return []

# Load all keywords for a cluster
def load_keywords_for_cluster(cluster_id, cluster_directory='dataset/clusters'):
    path = os.path.join(cluster_directory, f'cluster_{cluster_id}.xml')
    if not os.path.exists(path):
        print(f"[x] Couldn't find cluster file {path}")
        return []

    all_keywords = []
    try:
        with open(path, 'r') as f:
            content = f.read()

        tree_ids = re.findall(r"<Tree>(.*?)<\/Tree>", content)
        for tree_id_str in tree_ids:
            tree_id = int(tree_id_str.strip())
            tree_keywords = load_keywords_from_tree(tree_id)
            if tree_keywords:
                all_keywords.append(tree_keywords)

        return all_keywords

    except Exception as e:
        print(f"[X] Failed to load cluster {cluster_id}: {e}")
        return []

def get_cluster_trees(cluster_path):
    with open(cluster_path, 'r') as f:
        content = f.read()
        tree_ids = re.findall(r"<Tree>(.*?)<\/Tree>", content)
        return tree_ids or []

def write_cluster_file(cluster_path, cluster_id, tree_ids):
    with open(cluster_path, 'w') as f:
        f.write(f'<Cluster id="{cluster_id}">\n')
        for tree_id in sorted(set(tree_ids), key=int):
            f.write(f'\t<Tree>{tree_id}</Tree>\n')
        f.write('</Cluster>\n')

# Load GloVe embeddings
print("[~] Loading GloVe model...")
word_vectors = api.load("glove-wiki-gigaword-300")
print("[✓] GloVe model loaded.")


# Compute max-avg cosine similarity distance
def compute_similarity(cluster_a_words, cluster_b_words):
    words_a = [word.lower() for word in chain.from_iterable(cluster_a_words)]
    words_b = [word.lower() for word in chain.from_iterable(cluster_b_words)]

    vecs_a = [word_vectors[word] for word in words_a if word in word_vectors]
    vecs_b = [word_vectors[word] for word in words_b if word in word_vectors]

    if not vecs_a or not vecs_b:
        return 1.0  # maximum distance

    sim_matrix = cosine_similarity(vecs_a, vecs_b)

    max_a = np.mean(np.max(sim_matrix, axis=1))
    max_b = np.mean(np.max(sim_matrix, axis=0))

    avg_sim = (max_a + max_b) / 2
    return 1 - avg_sim  # similarity → distance

# Load clusters
cluster_dir = 'dataset/clusters'
cluster_dic = {}

# Filter valid clusters with keywords
for file in os.listdir(cluster_dir):
    if file.endswith(".xml"):
        cluster_id = int(re.findall(r'\d+', file)[0])
        cluster_words = load_keywords_for_cluster(cluster_id, cluster_dir)
        if any(cluster_words):
            cluster_dic[cluster_id] = cluster_words

# Map cluster IDs to matrix indices
cluster_ids = sorted(cluster_dic.keys())
id_to_index = {cid: idx for idx, cid in enumerate(cluster_ids)}
n = len(cluster_ids)
similarity_matrix = np.zeros((n, n))

# Compute pairwise distances
for i_id in tqdm(cluster_ids, desc="Computing similarity"):
   
    for j_id in cluster_ids:
        i = id_to_index[i_id]
        j = id_to_index[j_id]
        similarity_matrix[i][i] = 0
        if i <= j:  # avoid duplicate computations
            score = compute_similarity(cluster_dic[i_id], cluster_dic[j_id])
            similarity_matrix[i][j] = similarity_matrix[j][i] = score

plt.figure(figsize=(14, 12))
plt.imshow(similarity_matrix, cmap='viridis_r', interpolation='nearest')
plt.colorbar(label='Normalized Distance (0-1, lower = more similar)')
plt.title('Keyword Distance Matrix by Cluster ID')
plt.tight_layout()
plt.savefig('keyword_distance_matrix_normalized.png')
plt.show()

# Apply hierarchical clustering using the normalized matrix
# Convert distance matrix to condensed form for linkage
condensed_dist = squareform(similarity_matrix)
Z = linkage(condensed_dist, method='average')  # Use average linkage

# Threshold for defining cluster similarity (tuneable)
distance_threshold = 0.1  
clusters = fcluster(Z, distance_threshold, criterion='distance')

# Map resulting hierarchical clusters back to original cluster IDs
cluster_mapping = defaultdict(list)
for idx, group_id in enumerate(clusters):
    original_cluster_id = cluster_ids[idx]  # this fixes the issue!
    cluster_mapping[group_id].append(original_cluster_id)

# Show only merged clusters (with more than one cluster ID)
mergeable_clusters = {k: v for k, v in cluster_mapping.items() if len(v) > 1}

total_cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith(".xml")]
total_number_of_clusters = len(total_cluster_files)

# --- Only clusters with usable keywords (that are candidates for merging) ---
processed_cluster_count = len(cluster_ids)

# --- Compute number of merged clusters (i.e. reductions) ---
reduction = sum(len(ids) - 1 for ids in mergeable_clusters.values())

# --- Final cluster count after merging ---
final_cluster_count = total_number_of_clusters - reduction

# --- Report ---
reduction_percentage = (reduction / total_number_of_clusters) * 100 if total_number_of_clusters > 0 else 0

print("\n--- Final Report ---")
print(f"Total clusters (original): {total_number_of_clusters}")
print(f"Clusters with keywords (processed): {processed_cluster_count}")
print(f"Mergeable groups found: {len(mergeable_clusters)}")
print(f"Clusters reduced by: {reduction}")
print(f"Final cluster count: {final_cluster_count}")
print(f"Reduction percentage: {reduction_percentage:.2f}")

total_tree_merged = 0

for group_id, cluster_group in mergeable_clusters.items():
    print(f"Group {group_id} merging clusters: {cluster_group}")
    
    all_tree_ids = []

    # First cluster will be the one we keep
    target_cluster_id = cluster_group[0]
    target_cluster_path = os.path.join(cluster_dir, f'cluster_{target_cluster_id}.xml')

    for cluster_id in cluster_group:
        cluster_path = os.path.join(cluster_dir, f'cluster_{cluster_id}.xml')
        tree_ids = get_cluster_trees(cluster_path)
        all_tree_ids.extend(tree_ids)

    total_tree_merged += len(set(all_tree_ids))

    # Write the merged cluster to the target file
    write_cluster_file(target_cluster_path, target_cluster_id, all_tree_ids)

    # Remove the other cluster files
    for cluster_id in cluster_group[1:]:
        path_to_remove = os.path.join(cluster_dir, f'cluster_{cluster_id}.xml')
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove)
            print(f"Deleted redundant cluster: cluster_{cluster_id}.xml")

print(f"✅ Total unique trees merged: {total_tree_merged}")

     