# Hierarchical Clustering:

Unsupervised clustering algorithm which involves creating clusters that have predominant ordering from top to bottom. 
They fall into 2 categories: bottom-up (agglomerative) or top down (divisive). 

Bottom-up algorithms treat each data point as a single cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all data points. Bottom-up hierarchical clustering is therefore called hierarchical agglomerative clustering or HAC. 

This clustering technique is divided into two types:

    - 1. Agglomerative Hierarchical Clustering
    - 2. Divisive Hierarchical Clustering


## Agglomerative Hierarchical Clustering:
It's a “bottom-up” approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

How does it work?

    - 1. Make each data point a single-point cluster → forms N clusters
    - 2. Take the two closest data points and make them one cluster → forms N-1 clusters
    - 3. Take the two closest clusters and make them one cluster → Forms N-2 clusters.
    - 4. Repeat step-3 until you are left with only one cluster.
    
![Visual Representation](https://miro.medium.com/max/257/0*iozEcRXXWXbDMrdG.gif)

### Linkage Methods:

    1. Complete-linkage: the distance between two clusters is defined as the longest distance between two points in each cluster.
    2. Single-linkage: the distance between two clusters is defined as the shortest distance between two points in each cluster. This linkage may be used to detect high values in your dataset which may be outliers as they will be merged at the end.
    3. Average-linkage: the distance between two clusters is defined as the average distance between each point in one cluster to every point in the other cluster.
    4. Centroid-linkage: finds the centroid of cluster 1 and centroid of cluster 2, and then calculates the distance between the two before merging.


### Dendogram:

A Dendrogram is a type of tree diagram showing hierarchical relationships between different sets of data.
As already said a Dendrogram contains the memory of hierarchical clustering algorithm, so just by looking at the Dendrgram you can tell how the cluster is formed.

![Visual Dendogram](https://miro.medium.com/max/960/0*BfO2YN_BSxThfUoo.gif)

Note:-

    1. Distance between data points represents dissimilarities.
    2. Height of the blocks represents the distance between clusters.

#### Parts of Dendograms:

![Dendograms Parts](https://miro.medium.com/max/738/0*ESGWAWTMwZi_xTz-.png)

    - The Clades are the branch and are arranged according to how similar (or dissimilar) they are. Clades that are close to the same height are similar to each other; clades with different heights are dissimilar — the greater the difference in height, the more dissimilarity.
    - Each clade has one or more leaves.
    - Leaves A, B, and C are more similar to each other than they are to leaves D, E, or F.
    - Leaves D and E are more similar to each other than they are to leaves A, B, C, or F.
    - Leaf F is substantially different from all of the other leaves.

### Well this is all cool but when do we stop merging clusters then ? 

Ans. 
You cut the dendrogram tree with a horizontal line at a height where the line can traverse the maximum distance up and down without intersecting the merging point.

Ex.
![Visual Dendogram Cut Off](https://miro.medium.com/max/628/0*lO30pyuAmDk6h_0T.jpg)

L3 can traverse maximum distance up and down without intersecting the merging points. So we draw a horizontal line and the number of verticle lines it intersects is the optimal number of clusters.


## Divisive Hierarchical Clustering:

In Divisive or DIANA(DIvisive ANAlysis Clustering) is a top-down clustering method where we assign all of the observations to a single cluster and then partition the cluster to two least similar clusters. Finally, we proceed recursively on each cluster until there is one cluster for each observation. So this clustering approach is exactly opposite to Agglomerative clustering.

![DIANA](https://miro.medium.com/max/1010/0*OefsgEh-nRp5Rvm_.png)


## Measure the goodness of Clusters

#### Dunn's Index:

Dunn's index is simply a ratio between min intercluster distances to the maximum intra-cluster diameter.

The aim is to identify sets of clusters that are compact, with a small variance between members of the cluster, and well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster variance. For a given assignment of clusters, a higher Dunn index indicates better clustering.

Diameter of a cluster is simply the distance between its two futhermost point. We are looking for high Dunn's index.

 One of the drawbacks of using this is the computational cost as the number of clusters and dimensionality of the data increase.


## Thoughts/Conclusion:
Hierarchical clustering does not require us to specify the number of clusters and we can even select which number of clusters looks best since we are building a tree. Additionally, the algorithm is not sensitive to the choice of distance metric; all of them tend to work equally well whereas with other clustering algorithms, the choice of distance metric is critical. A particularly good use case of hierarchical clustering methods is when the underlying data has a hierarchical structure and you want to recover the hierarchy; other clustering algorithms can’t do this. These advantages of hierarchical clustering come at the cost of lower efficiency, as it has a time complexity of O(n³), unlike the linear complexity of K-Means and GMM.

