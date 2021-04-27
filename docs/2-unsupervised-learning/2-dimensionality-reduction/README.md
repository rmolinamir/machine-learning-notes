# Dimensionality Reduction

There are a couple of different reasons why one might want to do dimensionality reduction. One is data compression, data compression not only allows us to compress the data and have it therefore use up less computer memory or disk space, but it will also allow us to speed up our learning algorithms. Another reason is visualization of the data, e.g. how to visualize a data set of many features in a 2 or 3 dimensional space?

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Data Compression](#data-compression)
- [Visualization](#visualization)
- [Principle Component Analysis (PCA) Problem Formulation](#principle-component-analysis-pca-problem-formulation)
- [Principle Component Analysis (PCA) Applications](#principle-component-analysis-pca-applications)
- [Principle Component Analysis (PCA) Algorithm](#principle-component-analysis-pca-algorithm)
- [Reconstruction from Compressed Representation](#reconstruction-from-compressed-representation)
- [Choosing the number of Principle Components](#choosing-the-number-of-principle-components)
- [Advice for Applying PCA](#advice-for-applying-pca)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Data Compression

Data Compression speeds up algorithms and reduces space used by data for them. But, what is dimensionality reduction?

Let's say you've collected many features - maybe more than you need. Can you "simplify" your data set in a rational and useful way? For example, a redundant two dimensional data set - different units for the same attribute (e.g. centimeters and inches), we reduce data to 1D from 2D.

![Data Compression Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Data-Compression%20Example%20I.png)

Data redundancy can happen when different teams are working independently, this often generates redundant data (especially if you don't control the data collection). Another example is helicopter flying - if we do a survey of pilots (where x<sub>1</sub> is pilot skill, x<sub>2</sub> is pilot enjoyment). These features may be highly correlated, so this correlation can be combined into a single attribute called aptitude (for example).

To perform dimensionality reduction, let's plot a line in the following example, and project each example into the line:

![Data Compression Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Data-Compression%20Example%20II.png)

Before x<sub>1</sub> was a 2D feature vector (X and Y dimensions), now we can represent x<sub>1</sub> as a 1D number (Z dimension).

This allows us to half the amount of storage, gives lossy compression with hopefully an acceptable loss. The loss above comes from the rounding error in the measurement.

Another example of 3D data to 2D:

![Data Compression Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Data-Compression%20Example%20III.png)

We can project all of the 3 dimensional vectors (or points) in a 2D plane within the 3 dimensional space, by computing the distance from the vectors to the plane and projecting them in said plane, and we end up with a 2D visualization of the data. This is similar to the line projection in the previous 2D to 1D example.

![Data Compression Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Data-Compression%20Example%20IV.png)

So we've now reduced our 3D vector to a 2D vector. In reality we'd normally try and do reduce, for example, 1000D to 100D.

## Visualization

It's hard to visualize highly dimensional data. Dimensionality reduction can improve how we display information in a tractable manner for human consumption. It often helps to develop algorithms if we can understand our data better, so dimensionality reduction helps us to see the data in a way that can be more easily interpreted. It's good for explaining something to someone if you can "show" it in the data.

For example, we collect a large data set about many facts of a country around the world:

![Visualization Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Visualization-Example%20I.png)

Where:

- x<sub>1</sub>: GDP
- ...
- x<sub>6</sub>: Mean household income

Let's say that we have 50 features per country, how can we understand this data better? It's very hard to plot 50 dimensional data. Using dimensionality reduction, instead of each country being represented by a 50-dimensional feature vector, we can come up with a different feature representation (z values) which summarize these features.

![Visualization Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Visualization-Example%20II.png)

This gives us a 2 dimensional vector, where we reduced the data from 50D to 2D. And we can plot the result. Typically you don't generally ascribe meaning to the new features (so we have to determine what these summary values mean) e.g. you may find horizontal axis corresponds to overall country size/economic activity and that the y axis may be the per-capita well being/economic activity. So despite having 50 features, there may be two "dimensions" of information, with features associated with each of those dimensions.

It's up to you on how to assess in which way the features can be grouped to form summary features, and how best to do that (feature scaling is very important). This helps in showing the two main dimensions of variation in a way that's easy to understand.

![Visualization Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Visualization-Example%20III.png)

## Principle Component Analysis (PCA) Problem Formulation

To perform dimensionality reduction, the most commonly used algorithm is PCA. Say we have a 2D data set which we wish to reduce to 1D:

![Principle Component Analysis (PCA) Problem Formulation Example I](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[7].png)

How can we find a single line onto which to project this data? How do we determine this line?

PCA will compute a line where the distance between each point and the projected version will overall be as small as possible (blue lines below are short). In other words, PCA tries to find a lower dimensional surface so the sum of squares onto that surface is minimized. The blue lines are sometimes called the projection error, PCA tries to find the surface (a straight line in this case) which has the minimum projection error.

![Principle Component Analysis (PCA) Problem Formulation Example II](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[8].png)

You **should always do mean normalization and feature scaling** on your data before PCA so that the features x<sub>1</sub> and x<sub>2</sub> should have zero mean, and should have comparable ranges of values.

For a more formal description, to reduce 2D to 1D, we must find a vector u<sup>(1)</sup> onto which you can project the data so as to minimize the projection error. u<sup>(1)</sup> can be positive or negative -u<sup>(1)</sup> which makes no difference. Each of the vectors define the same red line.

![Principle Component Analysis (PCA) Problem Formulation Example III](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[9].png)

In a more general case, to reduce from N dimensions to K dimensions, we find K vectors (u<sup>(1)</sup>, u<sup>(2)</sup>, ... u<sup>(K)</sup>) onto which to project the data to minimize the projection error. There will be a large amount of vectors onto which we project the data. With this set of vectors, we project the data onto the subspace spanned by that set of vectors.

For example, to reduce 3D to 2D:

- Find a pair of vectors which define a 2D plane (surface) onto which you're going to project your data.

![Principle Component Analysis (PCA) Problem Formulation Example IV](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[10].png)

It's worth noting that **PCA is not linear regression**. Despite cosmetic similarities, they are very different. For linear regression, we fit a straight line to minimize the distance between a point and this line, but this distance is a VERTICAL distance, because the line describes the behavior of the features vs. the cost.

PCA minimizes the magnitude of the shortest **orthogonal distance**, these are very different things. More generally with linear regression we're trying to predict "y". With PCA there is no "y" - instead we have a list of features and all features are treated equally.

## Principle Component Analysis (PCA) Applications

1. Compression:
     - Reduce memory/disk needed to store data and speed up learning algorithm.
     - K is chosen by the ratio of the retained variance relative to the original data set.
2. Visualization:
   - Typically chose K = 2 or K = 3, because we can plot and visualize these values. It would be hard to plot and visualize a 4 (or more) dimensional space.
   - One thing often done wrong regarding PCA is using PCA to prevent over-fitting. If we have x<sup>i</sup> we have N features, z<sup>i</sup> has K features, with K < N. So, if we only have K features then maybe we're less likely to over fit? This might work well, but it is not a good way to address over-fitting. It's better to use regularization. This is because PCA "throws" away data when compressing without knowing what values are being lost. It will probably be fine if you're keeping most of the data, but if you're throwing away some crucial data then it's bad. So, you would need a 95% to 99% variance retained, whileas regularization will give you AT LEAST a way that is AT LEAST as good to solve over-fitting
   - A second PCA misuse, is when it's used to design the ML system with PCA from the start. But, what if you did the whole thing without PCA? Measure how does the system performs without PCA, and only perform PCA if you have a reason to believe it will help. PCA is easy enough to include as an add-on, e.g. as a processing step. Try without PCA first!

## Principle Component Analysis (PCA) Algorithm

Before applying PCA must do data pre-processing. Given a set of <i>m</i> unlabeled examples we must perform mean normalization. We replace each x<sub>j<sup>i</sup></sub> with x<sub>j</sub> - μ<sub>j</sub>. In other words, we determine the mean of each feature set, and then for each feature subtract the mean from the value, so we re-scale the mean to be 0 scale. If features have very different scales then we should scale them so that they all have a comparable range of values, e.g. x<sub>j<sup>i</sup></sub> is set to <i>(x<sub>j</sub> - μ<sub>j</sub>) / s<sub>j</sub></i>.
Where s<sub>j</sub> is some measure of the range, such as:

- Range (<i>biggest - smallest</i>).
- More commonly, the standard deviation.

The different features have very different scales. So for example, if x<sub>1</sub> is the size of a house, and x<sub>2</sub> is the number of bedrooms, we then also scale each feature to have a comparable range of values.

With data pre-processing done, PCA finds the lower dimensional sub-space which minimizes the sum of the square. In summary, for 2D -> 1D we'd be doing something like this:

![Principle Component Analysis (PCA) Algorithm Example I](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[11].png)

We need to compute two things:

- The u (unitary) vectors.
- Compute the z vectors.

The z vectors are the new lower dimensionality feature vectors. The mathematical derivation for the u vectors is very complicated, but once you've done it, the procedure to find each u vector is not as hard.

To reduce data from N-dimensional space to a K-dimensional space:

- First, compute the covariance matrix:
  ![Principle Component Analysis (PCA) Algorithm Covariance Matrix](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[12].png)

  - This is commonly denoted as Σ (greek upper case sigma), not a summation symbol.
  - This is an [n x n] matrix.
    - Remember that x<sub>i</sub> is a [n x 1] matrix, or a vector.
  - We can implement this as follows:
  ![Principle Component Analysis (PCA) Algorithm Covariance Matrix II](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[13].png)
- Compute the eigenvectors ~~<sup>(I don't know what the fuck these really mean)</sup>~~ of matrix Σ:
  - Computing these eigenvectors is done by:
    - `[U, S, V] = svd(sigma)`, where:
      - `svd` stands for singular value decomposition.
      - `svd` is more numerically stable than `eig`.
      - `eig` also computes the eigenvector.
      - `eig(sigma) = svd(sigma)`
      - U,S and V are matrices.
        - U matrix is also an [n x n] matrix.
        - **Turns out that the columns of U are the u vectors we want. So to reduce a system from N-dimensions to K-dimensions, we just take the first K-vectors from U (first K columns)**.
        ![Principle Component Analysis (PCA) Algorithm Covariance Matrix III](https://www.holehouse.org/mlclass/14_Dimensionality_Reduction_files/Image%20[14].png)
- After computing U, we need to find some way to project our N dimensional space (x) to a K dimensional space (z):
  - Take first K columns from the U matrix and stack them in columns forming a new [n x k] matrix - we'll call this this U<sub>reduce</sub>.
  - We calculate z as follows:
  - <i>z = U<sub>reduce</sub><sup>T</sup> * x</i>
    - A [k x n] \* [n x 1] matrix multiplication, which generates a matrix (or vector) which is [k x 1].  ~~<sup>(I still don't know what the fuck is happening)</sup>~~

In summary:

- Perform data-preprocessing.
- Calculate Σ (sigma, the covariance matrix).
- Calculate eigenvectors with `svd`.
- Take K vectors from U (U<sub>reduce</sub> = U[:, :k]).
- Calculate z (<i>z = U<sub>reduce</sub><sup>T</sup> * x</i>).

## Reconstruction from Compressed Representation

PCA works as a compression algorithm. f this is the case, is there a way to decompress the data from low dimensionality back to a higher  (or the original) dimensionality format?

Say we have an example as follows:

![Reconstruction from Compressed Representation Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Reconstruction-from%20Compressed%20Representation%20Example%20I.png)

- We have our examples (x<sup>1</sup>, x<sup>2</sup> etc.).
- Project onto z-surface.
- Given a point z<sup>1</sup>, how can we go back to the 2D space?

Considering that <i>z = U<sub>reduce</sub><sup>T</sup> * x</i>, we can calculate an approximation of x given by:

- <i>x<sub>approx</sub> = U<sub>reduce</sub><sup>T</sup> * z</i>

This would project the values of the K dimensional space back onto the unitary vector(s), in this case onto the line, and thus, this creates the following representation:

![Reconstruction from Compressed Representation Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Reconstruction-from%20Compressed%20Representation%20Example%20I.png)

We lose some of the information (i.e. everything is now perfectly on that line) but it is now projected back into the 2D space. The lost information can be measured by a number called *variance*.

## Choosing the number of Principle Components

How do we chose K? K is referred to as the number of principle components. To chose K, let's think about how PCA works. PCA tries to minimize the averaged squared projection error:

![Choosing the number of Principle Components Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Choosing-the%20number%20of%20Principle%20Components%20Example%20I.png)

And the average square sum of our data:

![Choosing the number of Principle Components Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Choosing-the%20number%20of%20Principle%20Components%20Example%20II.png)

The total variation in the data can be defined as the average squared projection error over average the sum of our data, denoting how far are the training examples from the origin. Therefore:

![Choosing the number of Principle Components Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Choosing-the%20number%20of%20Principle%20Components%20Example%20III.png)

This is the ratio between averaged squared projection error in relation to the total variation in the data. We want the ratio to be small - less than 0.01 (or 1%) means that we retain 99% of the variance.

If the ratio is small (close to 0), then this is because the numerator is small. The numerator is small when x<sup>i</sup> = x<sub>approx</sub><sup>i</sup>. In other words, we lose very little information in the dimensionality reduction, so when we decompress we regenerate the almost the same data.

So, we chose K in terms of this ratio. We can often significantly reduce data dimensionality while retaining the variance.

![Choosing the number of Principle Components Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/2-unsupervised-learning/2-dimensionality-reduction/images/Choosing-the%20number%20of%20Principle%20Components%20Example%20IV.png)

## Advice for Applying PCA

We can use PCA to speed up algorithm running time. Say you have a supervised learning problem with input x and y:

- x is a 10 000 dimensional feature vector, e.g. 100 x 100 images = 10000 pixels, such a huge feature vector will make the algorithm slow.
- With PCA we can reduce the dimensionality and make it tractable by:
  - Extract the x feature set data, this results in a unlabeled training set.
  - Apply PCA to x vectors, this results in a reduced dimensional feature vector z, and the z vector would become the new training set. Each vector can be re-associated with the label.
  - Take the reduced dimensionality data set and feed it to a learning algorithm (using y as labels and z as feature vector).

PCA maps one vector to a lower dimensionality vector, i.e. x -> z. The feature vector z is defined by PCA only on the training set. The mapping computes a set of parameters, such as the feature scaling values and more importantly, U<sub>reduce</sub>. This U<sub>reduce</sub> parameter is learned by performing PCA on the data set, and it's used on the cross validation data set and test data set, **it is not computed again**. Typically, you can reduce data dimensionality by 5 to 10 times without a major hit to the accuracy of the model.
