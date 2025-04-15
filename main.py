import os
import os.path
import pandas as pd
from Tree import TreeNode  , Tree , load_tree_from_file , save_tree_to_file
    
## function used to process and extract the urls from the archive
def unpack_dataset(directory_path , dataset_name):

    ## construct the path of the target file
    dataset_path = os.path.join(directory_path , dataset_name)
    
    ## check if the file has been found
    if not os.path.exists(dataset_path):
        print("Couldn't find dataset file!")
        return []
    
    #extracting the data frames
    frames = pd.read_parquet(dataset_path, engine='fastparquet')
    
    print(f'Extracted Dataset columns : {frames.columns}')
    print(f'Extracted Dataset size : {len(frames)}')

    return frames

def append_to_tree(url , root , tree_dictionary):
    tree = tree_dictionary[root]
    tree.insert_route(url)

def create_tree(url , root , tree_dictionary):
    tree = Tree()
    tree.insert_route(url)
    tree_dictionary[root] = tree
    

def persist_tree(root , tree_dictionary):
    file_path = os.path.join('dataset','trees' , f'{root}.tree')
    save_tree_to_file(file_path , tree_dictionary[root])


def process_url(url , tree_dictionary):
    
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
        append_to_tree(url , head , tree_dictionary)
    else:
        create_tree(url , head , tree_dictionary)

    persist_tree(head , tree_dictionary)
    


#to store each computed tree
tree_dictionary  = {}

dataset_folder = 'dataset/original'
dataset_name = 'logos.snappy (1).parquet'

frames = unpack_dataset(dataset_folder , dataset_name)

#sorting them to be processed in order
frames = frames.sort_values(by="domain")

tree_directory = 'dataset/trees'

if os.path.exists(tree_directory):
    for file in os.listdir(tree_directory):
        os.remove(os.path.join(tree_directory,file))

for frame in frames.values :
    print(frame[0])
    process_url(frame[0], tree_dictionary)

list = []
for key , value in tree_dictionary.items():
       list+=value.get_all_routes() 

for url in list:
    print(url)

# ========== TREE VERIFICATION ==========

print("\n[Verification] Comparing saved trees with in-memory trees...\n")

for root, original_tree in tree_dictionary.items():
    file_path = os.path.join(tree_directory, f"{root}.tree")
    loaded_tree = load_tree_from_file(file_path)

    if not loaded_tree:
        print(f"[✗] Failed to load tree from {file_path}")
        continue

    original_routes = set(original_tree.get_all_routes())
    loaded_routes = set(loaded_tree.get_all_routes())

    if original_routes == loaded_routes:
        print(f"[✓] Tree match for root '{root}'")
    else:
        print(f"[✗] Tree mismatch for root '{root}'")
        print(f"    - Original Routes: {len(original_routes)}")
        print(f"    - Loaded Routes:   {len(loaded_routes)}")
        print(f"    - Diff: {original_routes.symmetric_difference(loaded_routes)}")

    
    