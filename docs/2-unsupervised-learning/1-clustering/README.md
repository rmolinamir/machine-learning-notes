# Clustering

A supervised learning problem is given a set of labels to fit a hypothesis to it. In contrast, in unsupervised learning we're given data that does not have any labels associated with it. What we do is we give this unlabeled training set to an algorithm and we just ask the algorithm to find some structure in the data for us. One type of structure we might have an algorithm find is that it looks like a data set grouped into two separate clusters, and so an algorithm that finds clusters like these is called a clustering algorithm.

![Clustering Example](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[2].png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

- [What is Clustering good for?](#what-is-clustering-good-for)
- [K-Means Algorithm](#k-means-algorithm)
- [K-Means Algorithm for Non-Separated Clusters](#k-means-algorithm-for-non-separated-clusters)
- [K-Means Algorithm Optimization Objective](#k-means-algorithm-optimization-objective)
- [Random Initialization](#random-initialization)
- [How do we choose the Number of Clusters?](#how-do-we-choose-the-number-of-clusters)
  - [Elbow Method](#elbow-method)
  - [Downstream Purpose Method](#downstream-purpose-method)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## What is Clustering good for?

**One use case is market segmentation**, where you may have a database of customers and want to group them into different marker segments so you can sell to them separately or serve your different market segments better.

**Social network analysis** is another. There are actually groups that have done these things like looking at a group of people's social networks. So, companies like Facebook, etc., employ this. The clustering algorithm is used in a way where you know want to find who are the coherent groups of friends in the social network.

**Using clustering to organize computer clusters** or to organize data centers better. Because if you know which computers in the data center in the cluster tend to work together, you can use that to reorganize your resources and how you layout the network and how you design your data center communications.

And lastly, using clustering algorithms to **understand galaxy formations** and using that to understand astronomical data.

## K-Means Algorithm

In the clustering problem we are given an unlabeled data set and we would like to have an algorithm automatically group the data into coherent subsets or into coherent clusters for us. The K-Means algorithm is by far the most popular, and by far the most widely used clustering algorithm. The K-Means clustering algorithm is best illustrated in pictures. Let's say I want to take an unlabeled data set like the one shown here, and I want to group the data into two clusters.

![K-Means Algorithm Example I](https://www.holehouse.org/mlclass/13_Clustering_files/Image.png)

The first step is the Cluster assignment step. Here we randomly initialize two points, called the cluster centroids. So, these two crosses in the picture below are called the Cluster Centroids, and these centroids are randomly allocated points, you can have as many cluster centroid as you want (K cluster centroids). In this example, we have just two clusters.

![K-Means Algorithm Example II](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[1].png)

In this first step, we go through each example and depending on whether it's closer to the red or blue centroid, we assign each point to one of the two clusters. To demonstrate this, we've gone through the data and denote each point with red or blue respective to their assigned centroid (crosses) in the picture above.

In the second step we move the centroids, the "move centroids step". We take each centroid and move them to the average of the correspondingly assigned data-points.

![K-Means Algorithm Example III](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[2].png)

After moving the centroids, these steps are repeated until convergence. More technically, we can define this algorithm as having the following input and logic:

**Input:**

- K (number of clusters in the data)
- Training set {x<sup>1</sup>, x<sup>2</sup>, x<sup>3</sup> ..., x<sup>n</sup>).

**Algorithm:**

- Randomly initialize K cluster centroids as {μ<sub>1</sub>, μ<sub>2</sub>, μ<sub>3</sub> ..., μ<sub>K</sub>}
- Until convergence, the following loops are executed:
  - First inner loop:
    - This inner loop repeatedly assigns the c<sup>(i)</sup> variable to the index of the closest cluster centroid to x<sup>i</sup>. In other words, take i<sup>th</sup> example, measure the squared distance to each cluster centroid, then assign c<sup>(i)</sup> to the closest cluster centroid.
  - Second inner loop:
    - In this second loop, over each centroid calculate the average mean based on all the points associated with each centroid from c<sup>(i)</sup>.
  - If there is a centroid with no assigned data, either remove that centroid so we end up with K-1 cluster centroids, or randomly reinitialize.

![K-Means Algorithm Example IV](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[3].png)
![K-Means Algorithm Example V](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[4].png)

## K-Means Algorithm for Non-Separated Clusters

Very often the K-Means Algorithm is applied to datasets where there aren't well defined clusters, e.g. T-shirt sizes.

![K-Means Algorithm Non-Separated Clusters Example I](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[5].png)

These are not so obvious discrete groups. Say you want to have three sizes (small, medium, large - so <i>K = 3</i>), how big do we make these groups? One way would be to run K-Means on this data, so the following may be done:

![K-Means Algorithm Non-Separated Clusters Example II](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[6].png)

K-Means will "create" these three clusters, even though they aren't really there. By looking at the first population of people, we can design a small T-shirt which fits the 1st cluster, and so on for the other two. This is an example of market segmentation.

## K-Means Algorithm Optimization Objective

We know that supervised learning algorithms have an optimization objective (cost function). Just as these supervised learning algorithms, so does the K-Means algorithm. Using the following notation we can write the optimization objective:

![K-Means Algorithm Optimization Objective Example I](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[7].png)

K-Means has an optimization objective like the supervised learning functions we've seen. This is good because knowing because it helps for debugging, and it helps in finding the optimal clusters.

While K-Means is running, we keep track of two sets of variables:

- c<sup>(i)</sup> is the index of clusters {1,2, ..., K} to which x<sup>(i)</sup> is currently assigned. There are <i>m</i> c<sup>(i)</sup> values, as each example has a c<sup>(i)</sup> value, and that value is assigned to one the the clusters (i.e. can only be assigned to one of K different values).
- μ<sub>K</sub>, is the cluster associated with centroid K. These the centroids which exist in the training data space and are determined by their assigned x<sup>i</sup> examples. Additionally, μ<sub>c<sup>(i)</sup></sub> is the cluster centroid of the cluster to which the example x<sup>i</sup> has been assigned to, this is more for convenience than anything else.
  - You could look up that example <i>i</i> is indexed to cluster <i>j</i> (using the c vector), where <i>j</i> is between 1 and K. Then look up the value associated with cluster <i>j</i> in the <i>μ</i> vector (i.e. what are the features associated with μ<sub>j</sub>). But instead, for practicality, we create this variable μ<sub>c<sup>(i)</sup></sub> which stores the same value.

Lets say x<sup>i</sup> as been assigned to cluster 5. This means that:

- c<sup>(i)</sup> = 5
- μ<sub>c<sup>(i)</sup></sub> = μ<sub>5</sub>

So the cost are the squared distances between training examples x<sup>i</sup> and the cluster centroid to which x<sup>i</sup> has been assigned to. This is just what we've been doing, as the visual description below shows.

![K-Means Algorithm Optimization Objective Example II](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[8].png)

The red line here shows the distances between the example x<sup>i</sup> and the cluster to which that example has been assigned, when the example is close to the cluster, this value is small as it denotes the distance. When the cluster is very far away from the example, the value is large. This is sometimes called the **distortion** (or **distortion cost function**), so we are finding the values which minimizes this function.

![K-Means Algorithm Optimization Objective Example III](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[9].png)

If we consider the K-Means algorithm:

- The cluster assigned step is minimizing J(...) with respect to c<sup>(1)</sup>, c<sup>(2)</sup> ... c<sup>(i)</sup>, i.e. find the centroid closest to each example. Doesn't change the centroids themselves
- The move centroid step computes the values of μ which minimizes J(...) with respect to μ<sub>K</sub>.

So, we're partitioning the algorithm into two parts. The first part which minimizes the <i>c</i> variables, and the second part that minimizes the cost. We can use this knowledge to help debug our K-Means algorithm.

## Random Initialization

How can we initialize K-Means and avoid local optima? Consider the clustering algorithm. How do we actually initialize the centroids? There are a few ways, but it turns out that there is one method in particular that is the most recommended.

Have number of centroids set to less than the number of examples (K < m, because if K > m then we have a problem), then randomly pick K training examples.

When running K-Means, you should have the number of cluster centroids, K, set to be less than the number of training examples M. It would be really weird to run K-Means with a number of cluster centroids that's equal or greater than the number of examples you have, considering we're trying to find clusters within the data.

So, we set μ<sub>1</sub> up to μ<sub>K</sub> to the values of these examples. K-Means can converge to different solutions depending on the initialization setup due to a risk of the centroids converging in a local optima as shown in the picture below because the centroids are randomly initialized to be values of x<sup>i</sup>.

![Random Initialization Example I](https://www.holehouse.org/mlclass/13_Clustering_files/Image%20[10].png)

The local optima are valid convergence points, but local optima not global optima. If this is a concern, then we can do multiple random initializations to see if we get the same results, therefore many of these same results are likely to indicate that there is a global optima.

Concretely, here's how you could go about avoiding local optima. Let's say, I decide to run K-Means a hundred times so I'll execute this loop a hundred times, this being a fairly typical a number of times (usually it's a value from 50 up to may be 1000).

- For 100 iterations:
  - Randomly initialize K-Means.
  - Run K-Means Algorithm and compute c<sup>(1)</sup>, c<sup>(2)</sup> ... c<sup>(i)</sup>, μ<sub>1</sub>, ..., μ<sub>K</sub>.
- End with 100 results of clustered data, each with a distortion value (the cost).
- Pick the clustering which gave the lowest distortion.

Running K-Means with a range of K clusters can also help find better global optimum, e.g. K between 2 to 10. If K is larger than 10, then multiple random initializations are less likely to be necessary because it's less likely to make a huge difference and there is a much higher chance that your first random initialization will give you a pretty decent solution already. The first solution should probably be good enough (due to better granularity of clustering).

## How do we choose the Number of Clusters?

Unfortunately, there are no great ways to do this automatically because it's often generally ambiguous how many clusters there are in the data, and many times there might not be one right answer. And this is part of our supervised learning. We are aren't given labels, and so there isn't always a clear cut answer.

Normally we use **visualizations** to do it manually. When people talk about ways of choosing the number of clusters, one method that people sometimes talk about is something called the Elbow Method.

### Elbow Method

We're going to vary K, which is the total number of clusters. We're going to run K-Means with one cluster, so that everything gets grouped into a single cluster and compute the cost function or compute the distortion J and plot that here. And then we're going to run K means with two clusters, maybe with multiple random initial agents, maybe not. But then, you know, with two clusters we should get, hopefully, a smaller distortion, and so plot that there. And then run K-Means with three clusters, hopefully, you get even smaller distortion and plot that there.

Then I'm gonna run K-means with four, five and so on. And so we end up with a curve showing how the distortion, you know, goes down as we increase the number of clusters. And so we get a curve that maybe looks like the picture below.

![Elbow Method Example I](http://www.holehouse.org/mlclass/13_Clustering_files/Image%20[12].png)

If you look at this curve, what the Elbow Method does is say "Looks like there's a clear elbow there". You find this sort of pattern where the distortion goes down rapidly from 1 to 2, and 2 to 3, and then you reach an elbow at 3, and then the distortion goes down very slowly after that.

Then it looks like, maybe using three clusters is the right number of clusters, because the distortion goes down rapidly until K equals 3.

It turns out the Elbow Method isn't used that often, and one reason is that, if you actually use this on a clustering problem it turns out that fairly often, you end up with a curve that looks much more ambiguous that does not follows the above pattern (i.e. no clear elbow). This makes it harder to choose a number of clusters using this method.

### Downstream Purpose Method

Another method is to run K-Means with a later/downstream purpose and see how well a different number of clusters serve your later needs, such as using K-Means for market segmentation.

For example, T-shirt sizes. If you have three sizes (small, medium, large) or five sizes (extra small, small, medium, large, extra large), then run K means where K = 3 and K = 5. This would look like this:

![Downstream Purpose Method Example I](http://www.holehouse.org/mlclass/13_Clustering_files/Image%20[13].png)

This provides us with a way to chose the number of clusters. We could consider the cost of making extra sizes vs. how well distributed the products are. How important are those sizes though? (e.g. more sizes might make the customers happier). So, the applied problem may help guide the number of clusters.

For the most part, the number of customers K is still chosen by hand by human input or human insight. One way to try to do so is to use the Elbow Method, but I wouldn't always expect that to work well, but I think the better way to think about how to choose the number of clusters is to ask, for what purpose are you running K-means?
