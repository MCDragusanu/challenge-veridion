## How to run

1. Clone the project from github.
2. Create the Envirovment (tested on .venv)
3. Run `pip fastparquet lxml requests pandas beautifulsoup4 pandas numpy matplotlib tqdm scikit-learn scipy opencv-python pillow`

## Running Secquence
- Run each invidual stage in order

- stage_1.py -> stage_2.py -> stage_3.py -> stage_4.py

## Stage 1
- Stage 1 resets all the trees and logo features and re-builds the dataset. Modify the file `Logo.py` to alter what features are extracted. It prepares the trees and the vector features for each logo. Modify the `Scraper.py` to alter the scraping algorithm

## Stage 2
- Stage 2 resets all the prior clusterization and performes the similarity clustering to detect repeating logos for franchise etc. **Requires that the tree and logo directories to be populated.**

## Stage 3
- Stage 3 does the more advanced clustering usign the k-means algorithm. **It requires the cluster, trees & logos to be populated before running.**

## Stage 4
- Stage 4 is the one that outputs all the urls from clusters into dedicated text files for ease of use. Also has the option to download all the logos to be better for visualization.

- run with `--flush` to clear all the previos results

- run with `--no-logos` to skip the download of logos

- run with `--verbose` for debugging output file

**First Idea**

So my plan is to process the dataset and construct a tree / trie data structure to handle the routes that are 
from the same root website. With the processing i can cluster together all the logos that are related to the same
brand
               nike
            /   |   \
        shoes  sales  about-us
         /  \
      men  women


- All the logos that come from the children in this tree are related to the same brand / organisation and possibly for the same family of services / products.

- So for all the paths in the dataset i would build these trees, then for each website i would scrape it of the logo and some keywords / some information that can hint to the business sector of the organisation.Each such tree would define a class and all the logos that are in the same class, share the same 'features' let's call. So for each logo, besides the image-related features there will be also a class column pointing to the tree from which is part.

- So now i would say the problem is transformed into comparing two different classes atributes, and seeing how different they are. I'm thinking of using something like a **vector model**  to represent them into a geometric space and computing the dot product to check for similarities 

## Stage 1

**Step1 : Extracting and grouping the related urls**
- The archive is unpacked and sorted in asc order.Now the urls are processed and begin building the trees. After the trees are built they are issued an uid and are traversed and the urls are grouped together and stored in a xml file inside the `trees` directory, with the file name being uid of the tree and each route is assigned a custom uid.

**Step2 : Extracting logos and metadata**
-  For each page that url points to, will be scraped for logos and keywords information related to the domain.
-  There will be 2 parts of data for each route. Route related metadata like keywords for that page that was scraped and stored in it's coresponding `tree_id_.xml`. In the directory `dataset\logos\tree_{tree_uid}` will be saved all the features for the logo like this `logo_route_{route_id}_features.csv` so now each route is linked to it's logo and metadata features. For now will extract simple image features.
- All the images have been resized to fixed size 224 x 224 and basic features are extracted

**Step 1 & Step 2 Results and Observations**
- from 4384 unique routes ~1600 trees have been built so already some clusterization has been obtainen
1. maybe i can reduce even more the number of trees if i consider scenarios like for the trees 3 - 12. They are from the same mother company, but dedicated to a certain city `aamco.com` vs `aamcoantiochca.com`.Maybe if they share the same prefix the trees can be merged.
2. also many of these website use the same logo or very similar ones. So i would say the next step is to merge all the trees that have very similar logos.
- all the information necessary has been easeally grouped together for each route
- the structure is ready to begin the image processing and more complex features to be extracted 

## Stage 2

In stage 2 i will try and reduce the number of trees. My first idea is to handle the second point from above.I will make pair wise comparisons between logos of different trees. The simplest approach would be to calculate a `metric / distance` between the logo features of 2 trees.

Let's assume:

- Tree_X has {l_1_x^t .... l_n_x^t } logos where l_i_x is the feature vector of a logo of tree of id X

- Tree_Y has {l_1_y^t .... l_m_y^t } logos where l_i_y is the feature vector of a logo of tree of id Y

- similarity =avg (euclidian(l_i_x , l_j_y)). If m != n, will be computed with euclidian(l_i_x , 0)||euclidian(l_j_y , 0).

- In scenarios where 2 trees would be of size 1 and have similar images then similarity -> 0 and the trees can be merged. Will store somewhere that {X,Y,Z....} can be merged. 

- Matrix M_Similarity is a matrix where M_Similarity[i][j] = similarity score between Tree_i and Tree_j.So basically will store how different the trees are from each other, and maybe we can achieve even more clusterization where the similarity score is below some treshhold.

- This approach will also be flexible enough to add the keyword analisis to alter the similarity score when computed.
   
The clustering in this stage was performed solely on visual logo features, without incorporating contextual data such as associated keywords, business names, or descriptions. This isolates the analysis to branding appearance only.

**Stage 2 Result and Observations**

The features used for each logo were:

- Color statistics: mean_b, mean_g, mean_r (mean over the BGR color spectrum)

- Color variability: std_b, std_g, std_r (standard deviation across BGR)

- Shape/detail density: edge_density (estimated through edge detection metrics)

## Clustering Results: 

**Distance Recalculation** 
- Changed so that the distance is a normalized euclidian distance and clustering improved by 6%
- Original number of trees: 1580
- Distance threshold (normalized): 0.1
- Number of trees after merging: 592
- Reduction: 988 trees (62.53%)

- A heatmap of the normalized distance matrix revealed a very prominent cluster containing approximately 200 trees, primarily corresponding to AAMCO Services. This serves as a strong validation of the method, as identical or near-identical logos are correctly grouped.

- Other smaller but distinct clusters are also visible in the distance matrix visualization, suggesting the presence of additional branding reuse patterns across different business entities.

This method proves especially useful for:

- Detecting franchises and business chains that maintain consistent branding

- Uncovering logo duplication, template reuse, or common design elements

- Pre-processing before semantic or name-based clustering stages

- From what i can see it is very good at detecting templated logos , but fails to capture essential diferences, the primary grouping factor is the color and the writting usually i can spot logos similar in colors and writtings. Somehow there are also some similar shapes involved but it's to crude to detect more advanced features

A dedicated folder, dataset/clusters, has been created to store the resulting clusters. Each cluster is saved as an .xml file with its associated tree IDs, making the output easily interpretable and integrable with downstream processes.

## Stage 3
- In this stage, it is analyzed the semantic similarity between clusters of keywords, where each cluster consists of multiple trees and each tree contributes a list of keywords. To evaluate similarity across clusters, i use pre-trained GloVe embeddings (Global Vectors for Word Representation) to project keywords into a shared vector space.

- Similarity Methodology: Max-Avg Cosine Distance
It computes the distance between two keyword clusters using the max-avg cosine similarity technique:

- Embed all keywords from both clusters using GloVe vectors.

Compute pairwise cosine similarities between each keyword in cluster A and each in cluster B, resulting in a similarity matrix.

1. For each keyword in cluster A, take the maximum similarity it has with any keyword in cluster B. Repeat in the opposite direction (B → A).

2. Compute the average of all maximum similarities obtained in both directions.

3. Transform similarity to distance using: `distance = 1 - avg_max_similarity`


**Stage 3 Results**
- It doens't seem to be that effective, maybe because the websites are in different languages and the words are very different, also is pretty slow. 
- The keyword scraping isn't that effective because it s not very controllable in terms of the quality of the keywords, also it doen't take in consideration different languages of the websites.
- it still reduces the number of clusters by a tenth but still some more room of improvement.
- I think i should now focuse on better image clustering and feature extractions
- Original number of clusters: 371
- Distance threshold (normalized): 0.45
- Number of clusters after merging: 328
- Reduction: 43 clusters (11.59%)
- Number of mergeable groups: 22
- In some scenarios was able to group in the same cluster (cluster 40.txt i think) all websites that are about job adveritising which quite interesting, but not very effective overall, it should be more focused on the image processings

## Second Idea

-After analysing the clustering i concluded that the current logo features and similarity approach is effective for detecing templated and reapeating logos, which is great, but lacks the depth necessary to distinguigh more complex shapes, so far it is effective at grouping logos with similar colors

- So i will add more features related to positioning , center of mass, shapes, countours, color distributions ,  hues, to add more depth to the feature vector

- So the first issues to fix is to identify franchises and group them together. The goal would be to cluster together all the logos that have a very small similarity score. So i would use the stage 2 again but with a very small treshold to group all the logos together.

## Stage 3
- In stage 3 after each cluster has been formed, a `cluster representative` will be picked - the one that si the closest to the mean of cluster, the one that represents the best the features of the group.

- Then a K-Means clustering will be applied over the cluster representatives to make it more efficient , to minimize the number in the dataset that needs to be grouped toghether

- After this all the new formed clusters will be joined and saved in the datset for future stages if needed

**Result**

- It achieved a final clusterization of about 510 clusters which is good, considering that many clusters are made isolated trees, so about 200 such clusters are made of a single element and about 50 - 100 clusters of less than 10 - 15. 

- The similarity works great with colors and acceptable with shapes

-  In each of the cluster logo folder can be spoted leading features that dictate the rest of the style of the members , like color (red, blue , green) also shape (circular, square, etc). It can identify similar logos that contain that particular shape or emblem as well as letters. I have seen a lot of the same logo "type" many logos that have the same style (ex :  Some writting in the middle and surrounded by color), or even embedded shapes that respect the same coloring schema (ex : in some clusters all the logo have the same color and shape in the same area).

- There is still a lot of room of improvemnt in the image processing and image classification, it uses a generic k-means clustering tehnique. If i used a more advanced tehniques of feature extractions like using CNN or Ml to extract advanced features or doing advanced image preprocessing for cleaning the data it would have a significant impact

- Also i researched about PDQ which is algorithm created by Meta to compute image similarity fast and efficiently. I was very tempted to use it but i did some google searching and it seemed exaclty what i needed but didn't have time to implement it.
