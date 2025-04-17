import os
import os.path
import pandas as pd
from Tree import TreeNode  , Tree , load_tree_from_file , save_tree_to_file
from Scraper import scrape_site_info    
from Logo import get_bgr_matrix_from_url , extract_features_from_logo
import concurrent.futures
import pandas as pd
from itertools import count
#auto increment uids for trees and routes
tree_uid_counter = count(start=1)
route_uid_counter = count(start=1)

## function used to process and extract the urls from the archive
def unpack_dataset(directory_path , dataset_name):

    ## construct the path of the target file
    dataset_path = os.path.join(directory_path , dataset_name)
    
    print(dataset_path)
    ## check if the file has been found
    if not os.path.exists(dataset_path):
        print("Couldn't find dataset file!")
        return []
    
    #extracting the data frames
    frames = pd.read_parquet(dataset_path, engine='fastparquet')
    
    print(f'Extracted Dataset columns : {frames.columns}')
    print(f'Extracted Dataset size : {len(frames)}')

    return frames

def append_to_tree(url , root , tree_dictionary , route_id):
    tree = tree_dictionary[root]
    tree.insert_route(url , route_id)

def create_tree(url , root , tree_dictionary, route_id):
    global tree_uid
    tree_uid = next(tree_uid_counter)
    tree = Tree(treeId= tree_uid)
    tree.insert_route(url , route_id)
    tree_dictionary[root] = tree
   
def persist_tree(root , tree_dictionary, keywords , logo):
    tree  = tree_dictionary[root]
    file_path = os.path.join('dataset','trees' , f'tree_{tree.getId()}.xml')
    save_tree_to_file(file_path , tree, keywords , logo)

def store_logo_features(file_path, pd_frame):
    try:
        if pd_frame is not None:  
            pd_frame.to_csv(file_path, index=False)
            print(f"[✓] Data saved to: {file_path}")
        else:
            print("No Logo data found")
        return True
    except Exception as e:
        print(f"[Error] Could not save to CSV: {e}")
        return False
                
def process_url(url , tree_dictionary , logos_directory = 'dataset/logos/' ):
    

    route_id = next(route_uid_counter)

    if url is None or url == '':
        print(f'url ${url} is invalid!')
    #split by the /
    # a leaf node will be the .com in nike.com
    routes = url.replace('.' , '/').replace('-','/').split('/')

    if len(routes) == 1 or 0 :
        #TODO
        #think what you have to do in this scenario
        print(f'{url} is empty or no children')
        routes = url.split('.')
    
    head = routes[0]

    
    #each tree will have the name of the root element
    #a tree already exists
    if head in tree_dictionary :
        append_to_tree(url , head , tree_dictionary , route_id)
    else:
        create_tree(url , head , tree_dictionary , route_id)
    data = scrape_site_info(url)
    keywords = data["keywords"]
    logo = data["logo"]
    persist_tree(head , tree_dictionary , keywords , logo)
    current_tree = tree_dictionary[head]
    image = get_bgr_matrix_from_url(logo)
    features = extract_features_from_logo(image)

    folder_path = os.path.join(logos_directory , f'tree_{current_tree.getId()}')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path , mode= 777) 
    
    dest_path = os.path.join(folder_path , f'logo_route_{route_id}_features.csv')
    store_logo_features(dest_path , features)



#to store each computed tree
tree_dictionary  = {}

dataset_folder = 'dataset/original'
dataset_name = 'dataset.parquet'

frames = unpack_dataset(dataset_folder , dataset_name)

#sorting them to be processed in order
frames = frames.sort_values(by="domain")

tree_directory = 'dataset/trees'
logo_directory = 'dataset/logos'

#cleaning up the dataset directories
if os.path.exists(logo_directory):
    for file in os.listdir(logo_directory):
        os.rmdir(os.path.join(logo_directory,file))

if os.path.exists(tree_directory):
    for file in os.listdir(tree_directory):
        os.remove(os.path.join(tree_directory,file))

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []

    for frame in frames.values:
        url = frame[0]
        print(url)
        futures.append(executor.submit(process_url, url, tree_dictionary ))

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"[Error] Thread failed: {e}")


