**First Idea**

So my plan is to process the dataset and construct a tree / trie data structure to handle the routes that are 
from the same root website. With the processing i can cluster together all the logos that are related to the same
brand
                   nike
                /   \    \
            shoes   sales about-us
            /\
        man women .........

- All the logos that come from the children in this tree are related to the same brand / organisation and possibly for the same family of services / products.

- So for all the paths in the dataset i would build these trees, then for each website i would scrape it of the logo and some keywords / some information that can hint to the business sector of the organisation.Each such tree would define a class and all the logos that are in the same class, share the same 'features' let's call. So for each logo, besides the image-related features there will be also a class column pointing to the tree from which is part.

- So now i would say the problem is transformed into comparing two different classes atributes, and seeing how different they are. I'm thinking of using something like a **vector model**  to represent them into a geometric space and computing the dot product to check for similarities 

**Stage 1**

## Step1 : Extracting and grouping the related urls
- The archive is unpacked and sorted in asc order.Now the urls are processed and begin building the trees. After the trees are built they are issued an uid and are traversed and the urls are grouped together and stored in a xml file inside the `trees` directory, with the file name being uid of the tree and each route is assigned a custom uid.

## Step2 : Extracting logos and metadata
-  For each page that url points to, will be scraped for logos and keywords information related to the domain.
-  There will be 2 parts of data for each route. Route related metadata like keywords for that page that was scraped and stored in it's coresponding `tree_id_.xml`. In the directory `dataset\logos\tree_{tree_uid}` will be saved all the features for the logo like this `logo_route_{route_id}_features.csv` so now each route is linked to it's logo and metadata features. For now will extract simple image features.
- All the images have been resized to fixed size 224 x 224 and basic features are extracted

**Step 1 & Step 2 Results and Observations**
- from 4384 unique routes ~1600 trees have been built so already some clusterization has been obtainen
1. maybe i can reduce even more the number of trees if i consider scenarios like for the trees 3 - 12. They are from the same mother company, but dedicated to a certain city `aamco.com` vs `aamcoantiochca.com`.Maybe if they share the same prefix the trees can be merged.
2. also many of these website use the same logo or very similar ones. So i would say the next step is to merge all the trees that have very similar logos.
- all the information necessary has been easeally grouped together for each route
- the structure is ready to begin the image processing and more complex features to be extracted 

**Stage 2**

In stage 2 i will try and reduce the number of trees. My first idea is to handle the second point from above.I will make pair wise comparisons between logos of different trees. The simplest approach would be to check if the features.
Let's assume:
- Tree_X has {l_1_x^t .... l_n_x^t } logos where l_i_x is the feature vector of a logo of tree of id X
- Tree_Y has {l_1_y^t .... l_m_y^t } logos where l_i_y is the feature vector of a logo of tree of id Y
- similarity =Sum(euclidian(l_i_x , l_j_y)). If m != n, will be computed with euclidian(l_i_x , 0)||euclidian(l_j_y , 0).
- In scenarios where 2 trees would be of size 1 and have similar images similarity -> 0 and the trees can be merged. Will store somewhere that {X,Y,Z....} can be merged. 
- Matrix M_Similarity is a matrix where M_Similarity[i][j] = similarity score between Tree_i and Tree_j.So basically will store how different the trees are from each other, and maybe we can achieve even more clusterization where the similarity score is below some treshhold.
- This approach will also be flexible enough to add the keyword analisis to alter the similarity score when computed.
   