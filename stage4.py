import re
import os
import urllib.request
import urllib.parse
import hashlib
from urllib.error import URLError
from pathlib import Path

tree_directory = 'dataset/trees'
clusters_directory = 'dataset/clusters'
output_directory = 'results'
logo_directory = os.path.join(output_directory, 'logos')

def get_cluster_trees(cluster_path):
    with open(cluster_path, 'r') as f:
        content = f.read()
        tree_ids = re.findall(r"<Tree>(.*?)<\/Tree>", content)
        return tree_ids or []

def get_domain_and_logos_from_tree(tree_id, tree_directory='dataset/trees'):
    path = os.path.join(tree_directory, f'tree_{tree_id}.xml')
    if not os.path.exists(path):
        print(f"[x] Couldn't find {path} file")
        return [], []
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        domains = re.findall(r"<Domain>(.*?)<\/Domain>", content, flags=re.DOTALL)
        
        # Extract all logo URLs from routes
        logos = re.findall(r"<Logo>(.*?)<\/Logo>", content, flags=re.DOTALL)
        
        # Clean up the logos (remove whitespace)
        logos = [logo.strip() for logo in logos if logo.strip()]
        
        return domains, logos
    
    except Exception as e:
        print(f"[X] Failed to load tree {tree_id}: {e}")
        return [], []

def download_logo(logo_url, cluster_id, tree_id, logo_index=0):
    # Create a safe filename based on URL to ensure uniqueness
    url_hash = hashlib.md5(logo_url.encode()).hexdigest()[:8]
    
    # Get original extension if possible
    parsed_url = urllib.parse.urlparse(logo_url)
    filename = os.path.basename(parsed_url.path)
    extension = os.path.splitext(filename)[1]
    
    # If no valid extension, default to .ico
    if not extension:
        extension = '.ico'
    
    # Create unique filename with tree_id, logo_index and hash
    unique_filename = f"{tree_id}_{logo_index}_{url_hash}{extension}"
    
    # Create the cluster logo directory if it doesn't exist
    cluster_logo_dir = os.path.join(logo_directory, cluster_id)
    os.makedirs(cluster_logo_dir, exist_ok=True)
    
    save_path = os.path.join(cluster_logo_dir, unique_filename)
    
    try:
        # Skip if already downloaded (check if file exists)
        if os.path.exists(save_path):
            print(f"Logo already exists: {save_path}")
            return save_path
            
        urllib.request.urlretrieve(logo_url, save_path)
        print(f"Downloaded logo: {logo_url} to {save_path}")
        return save_path
    except URLError as e:
        print(f"Failed to download logo {logo_url}: {e}")
        return None
    except Exception as e:
        print(f"Error downloading logo {logo_url}: {e}")
        return None

# Create necessary directories
if os.path.exists(output_directory):
    # Only clear text files, not logo directories
    for file in os.listdir(output_directory):
        file_path = os.path.join(output_directory, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            os.remove(file_path)
else:
    os.mkdir(output_directory)

# Create logo directory
os.makedirs(logo_directory, exist_ok=True)

clusters = os.listdir(clusters_directory)

for cluster_file in clusters:
    cluster_path = os.path.join(clusters_directory, cluster_file)
    trees = get_cluster_trees(cluster_path)
    domains = []
    
    # Extract cluster_id correctly
    cluster_id = cluster_file.split('_')[-1].split('.')[0]
    
    # Create cluster logo directory
    cluster_logo_dir = os.path.join(logo_directory, cluster_id)
    os.makedirs(cluster_logo_dir, exist_ok=True)
    
    logo_count = 0
    
    for tree in trees:
        tree_domains, logos = get_domain_and_logos_from_tree(tree)
        domains.extend(tree_domains)
        
        # Download all logos for this tree
        for i, logo_url in enumerate(logos):
            if logo_url:
                download_logo(logo_url, cluster_id, tree, i)
                logo_count += 1
    
    # Write domains to output file
    output_path = os.path.join(output_directory, f'{cluster_id}.txt')
    with open(output_path, 'w') as f:
        for domain in domains:
            f.writelines(f'\n{domain}')
    print(f"Output file written: {output_path}")
    
    print(f"Processed cluster {cluster_id}: {len(domains)} domains, {logo_count} logos downloaded")

print("Processing complete!")