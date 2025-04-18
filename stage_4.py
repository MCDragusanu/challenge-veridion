import re
import os
import sys
import argparse
import urllib.request
import urllib.parse
import hashlib
import logging
from urllib.error import URLError
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import shutil
import time


# Configure logging
def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("logo_downloader.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("logo_downloader")


# Initialize logger
logger = setup_logging()


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Process clusters and download logos.')
    parser.add_argument('--flush', action='store_true', help='Flush current results and logos.')
    parser.add_argument('--no-logos', action='store_true', help='Skip logo downloads.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    return args


def flush_results(output_directory):
    """Delete all previous results and logos"""
    logger.info("Flushing previous results...")
    
    # Remove text files
    file_count = 0
    for file in os.listdir(output_directory):
        file_path = os.path.join(output_directory, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            os.remove(file_path)
            file_count += 1
    
    # Remove logo directories
    logos_path = os.path.join(output_directory, 'logos')
    if os.path.exists(logos_path):
        dir_count = 0
        for directory in os.listdir(logos_path):
            directory_path = os.path.join(logos_path, directory)
            if os.path.isdir(directory_path):
                try:
                    shutil.rmtree(directory_path)
                    dir_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove directory {directory_path}: {e}")
        
        try:
            os.rmdir(logos_path)
            logger.info(f"Flushed {file_count} text files and {dir_count} logo directories")
        except Exception as e:
            logger.error(f"Failed to remove logos directory: {e}")


def get_cluster_trees(cluster_path):
    """Extract tree IDs from a cluster file"""
    try:
        with open(cluster_path, 'r') as f:
            content = f.read()
            tree_ids = re.findall(r"<Tree>(.*?)<\/Tree>", content)
            logger.debug(f"Found {len(tree_ids)} trees in {os.path.basename(cluster_path)}")
            return tree_ids or []
    except Exception as e:
        logger.error(f"Failed to read cluster file {cluster_path}: {e}")
        return []


def get_domain_and_logos_from_tree(tree_id, tree_directory='dataset/trees'):
    """Extract domains and logo URLs from a tree file"""
    path = os.path.join(tree_directory, f'tree_{tree_id}.xml')
    if not os.path.exists(path):
        logger.warning(f"Tree file not found: {path}")
        return [], []
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        domains = re.findall(r"<Domain>(.*?)<\/Domain>", content, flags=re.DOTALL)
        logos = re.findall(r"<Logo>(.*?)<\/Logo>", content, flags=re.DOTALL)
        logos = [logo.strip() for logo in logos if logo.strip()]
        
        logger.debug(f"Tree {tree_id}: Found {len(domains)} domains and {len(logos)} logos")
        return domains, logos
    except Exception as e:
        logger.error(f"Failed to load tree {tree_id}: {e}")
        return [], []


def download_logo(logo_url, cluster_id, tree_id, logo_index=0, max_retries=1):
    """Download a logo from URL and save it to appropriate directory"""
    if not logo_url:
        logger.debug(f"Empty logo URL for tree {tree_id}, skipping")
        return None
        
    # Create a unique filename based on URL hash
    url_hash = hashlib.md5(logo_url.encode()).hexdigest()[:8]
    parsed_url = urllib.parse.urlparse(logo_url)
    filename = os.path.basename(parsed_url.path)
    extension = os.path.splitext(filename)[1]
    
    # Default to .ico if no extension found
    if not extension:
        extension = '.ico'
    
    unique_filename = f"{tree_id}_{logo_index}_{url_hash}{extension}"
    cluster_logo_dir = os.path.join(logo_directory, cluster_id)
    os.makedirs(cluster_logo_dir, exist_ok=True)
    save_path = os.path.join(cluster_logo_dir, unique_filename)

    # Skip if already downloaded
    if os.path.exists(save_path):
        logger.debug(f"Logo already exists: {save_path}")
        return save_path
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(logo_url, timeout=10)
            response.raise_for_status()
            
            # Verify it's a valid image
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Save the image
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Downloaded logo: {logo_url} to {save_path}")
            return save_path
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt+1}/{max_retries} failed for {logo_url}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to download logo after {max_retries} attempts: {logo_url}")
                return None
            time.sleep(1)  # Wait before retry
            
        except Exception as e:
            logger.error(f"Error processing logo {logo_url}: {e}")
            return None


def process_cluster(cluster_file, clusters_directory, tree_directory, output_directory, logo_directory, download_logos=True):
    """Process a single cluster file, extract domains and download logos"""
    cluster_path = os.path.join(clusters_directory, cluster_file)
    cluster_id = cluster_file.split('_')[-1].split('.')[0]
    
    logger.info(f"Processing cluster {cluster_id}")
    
    # Get trees in this cluster
    trees = get_cluster_trees(cluster_path)
    logger.info(f"Cluster {cluster_id} contains {len(trees)} trees")
    
    domains = []
    logo_count = 0
    
    # Create logo directory if needed
    if download_logos:
        cluster_logo_dir = os.path.join(logo_directory, cluster_id)
        os.makedirs(cluster_logo_dir, exist_ok=True)
    
    # Process each tree
    for tree in trees:
        tree_domains, logos = get_domain_and_logos_from_tree(tree, tree_directory)
        domains.extend(tree_domains)

        if download_logos:
            for i, logo_url in enumerate(logos):
                if logo_url:
                    result = download_logo(logo_url, cluster_id, tree, i)
                    if result:
                        logo_count += 1
    
    # Write domains to output file
    output_path = os.path.join(output_directory, f'{cluster_id}.txt')
    with open(output_path, 'w') as f:
        for domain in domains:
            f.write(f'{domain}\n')
    
    logger.info(f"Completed cluster {cluster_id}: {len(domains)} domains, {logo_count} logos downloaded")
    return len(domains), logo_count


def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Define paths
    tree_directory = 'dataset/trees'
    clusters_directory = 'dataset/clusters'
    output_directory = 'results'
    global logo_directory  # Make accessible to download_logo function
    logo_directory = os.path.join(output_directory, 'logos')
    
    # Log settings
    logger.info("Starting logo download process")
    logger.info(f"Settings: flush={args.flush}, download_logos={not args.no_logos}")
    
    # Flush results if requested
    if args.flush:
        flush_results(output_directory)
    
    # Prepare output directories
    os.makedirs(output_directory, exist_ok=True)
    if not args.no_logos:
        os.makedirs(logo_directory, exist_ok=True)
    
    # Get list of cluster files
    try:
        clusters = os.listdir(clusters_directory)
        logger.info(f"Found {len(clusters)} cluster files to process")
    except Exception as e:
        logger.error(f"Failed to list cluster directory {clusters_directory}: {e}")
        return
    
    # Process each cluster
    total_domains = 0
    total_logos = 0
    start_time = time.time()
    
    for i, cluster_file in enumerate(clusters):
        logger.info(f"Progress: {i+1}/{len(clusters)} ({(i+1)/len(clusters)*100:.1f}%)")
        domains, logos = process_cluster(
            cluster_file, 
            clusters_directory, 
            tree_directory, 
            output_directory, 
            logo_directory, 
            not args.no_logos
        )
        total_domains += domains
        total_logos += logos
    
    # Log summary
    elapsed_time = time.time() - start_time
    logger.info("Processing complete!")
    logger.info(f"Total domains: {total_domains}")
    logger.info(f"Total logos downloaded: {total_logos}")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()