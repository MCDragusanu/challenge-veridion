**First Idea**

So my plan is to process the dataset and construct a tree / trie data structure to handle the routes that are 
from the same root website. With the processing i can cluster together all the logos that are related to the same
brand
                   nike
                /   \    \
            shoes   sales about-us
            /\
        man women .........

-All the logos that come from the children in this tree are related to the same brand / organisation and possibly for the same family of services / products.

-So for all the paths in the dataset i would build these trees, then for each website i would scrape it of the logo and some keywords / some information that can hint to the business sector of the organisation.Each such tree would define a class and all the logos that are in the same class, share the same 'features' let's call. So for each logo, besides the image-related features there will be also a class column pointing to the tree from which is part.

-So now i would say the problem is transformed into comparing two different classes atributes, and seeing how different they are. I'm thinking of using something like a **vector model**  to represent them into a geometric space and computing the dot product to check for similarities 

