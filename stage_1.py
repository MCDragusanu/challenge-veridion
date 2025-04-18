import os
import shutil
import pandas as pd
from Tree import TreeNode, Tree, load_tree_from_file, save_tree_to_file
from Scraper import scrape_site_info    
from Logo import get_bgr_matrix_from_url, extract_features_from_logo
import concurrent.futures
from itertools import count

# Auto-incrementing IDs for trees and routes
tree_uid_counter = count(start=1)
route_uid_counter = count(start=1)

def unpack_dataset(directory_path, dataset_name):
    dataset_path = os.path.join(directory_path, dataset_name)
    print(f"[✓] Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print("[✗] Couldn't find dataset file!")
        return []

    frames = pd.read_parquet(dataset_path, engine='fastparquet')

    print(f'[✓] Dataset columns: {frames.columns}')
    print(f'[✓] Dataset size: {len(frames)}')

    return frames

def append_to_tree(url, root, tree_dictionary, route_id):
    tree = tree_dictionary[root]
    tree.insert_route(url, route_id)

def create_tree(url, root, tree_dictionary, route_id):
    tree_uid = next(tree_uid_counter)
    tree = Tree(treeId=tree_uid)
    tree.insert_route(url, route_id)
    tree_dictionary[root] = tree

def persist_tree(root, tree_dictionary, keywords, logo):
    tree = tree_dictionary[root]
    file_path = os.path.join('dataset', 'trees', f'tree_{tree.getId()}.xml')
    save_tree_to_file(file_path, tree, keywords, logo)

def store_logo_features(file_path, pd_frame):
    try:
        if pd_frame is not None:
            pd_frame.to_csv(file_path, index=False)
            print(f"[✓] Data saved to: {file_path}")
        else:
            print("[!] No Logo data found")
        return True
    except Exception as e:
        print(f"[Error] Could not save to CSV: {e}")
        return False

def process_url(url, tree_dictionary, logos_directory='dataset/logos/'):
    route_id = next(route_uid_counter)

    if url is None or url == '':
        print(f"[!] URL {url} is invalid!")
        return

    routes = url.replace('.', '/').replace('-', '/').split('/')
    if len(routes) <= 1:
        print(f"[!] {url} is too short, fallback to split by dot")
        routes = url.split('.')

    head = routes[0]

    if head in tree_dictionary:
        append_to_tree(url, head, tree_dictionary, route_id)
    else:
        create_tree(url, head, tree_dictionary, route_id)

    data = scrape_site_info(url)
    keywords = data["keywords"]
    logo = data["logo"]
    persist_tree(head, tree_dictionary, keywords, logo)

    current_tree = tree_dictionary[head]
    image = get_bgr_matrix_from_url(logo)
    features = extract_features_from_logo(image)

    folder_path = os.path.join(logos_directory, f'tree_{current_tree.getId()}')
    os.makedirs(folder_path, exist_ok=True)

    dest_path = os.path.join(folder_path, f'logo_route_{route_id}_features.csv')
    store_logo_features(dest_path, features)

# Main execution
tree_dictionary = {}

dataset_folder = 'dataset/original'
dataset_name = 'dataset.parquet'
tree_directory = 'dataset/trees'
logo_directory = 'dataset/logos'

# Create required directories if they don't exist
os.makedirs(tree_directory, exist_ok=True)
os.makedirs(logo_directory, exist_ok=True)

# Clean directories
for folder in os.listdir(logo_directory):
    full_path = os.path.join(logo_directory, folder)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)

for file in os.listdir(tree_directory):
    file_path = os.path.join(tree_directory, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Load dataset and process
frames = unpack_dataset(dataset_folder, dataset_name)
frames = frames.sort_values(by="domain")

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for frame in frames.values:
        url = frame[0]
        print(f"[→] Processing URL: {url}")
        futures.append(executor.submit(process_url, url, tree_dictionary))

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"[Error] Thread failed: {e}")
