# Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

Example: Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. The algorithm tries to find patterns within the chaos.

## Unsupervised Learning Model Representation

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h<sub>(x)</sub> is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Contents](#contents)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Contents

- [Clustering](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#clustering)
  - [What is Clustering good for?](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#what-is-clustering-good-for)
  - [K-Means Algorithm](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#k-means-algorithm)
  - [K-Means Algorithm for Non-Separated Clusters](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#k-means-algorithm-for-non-separated-clusters)
  - [K-Means Algorithm Optimization Objective](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#k-means-algorithm-optimization-objective)
  - [Random Initialization](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#random-initialization)
  - [How do we choose the Number of Clusters?](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#how-do-we-choose-the-number-of-clusters)
    - [Elbow Method](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#elbow-method)
    - [Downstream Purpose Method](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/1-clustering/#downstream-purpose-method)
- [Dimensionality Reduction](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#dimensionality-reduction)
  - [Data Compression](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#data-compression)
  - [Visualization](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#visualization)
  - [Principle Component Analysis (PCA) Problem Formulation](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#principle-component-analysis-pca-problem-formulation)
  - [Principle Component Analysis (PCA) Applications](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#principle-component-analysis-pca-applications)
  - [Principle Component Analysis (PCA) Algorithm](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#principle-component-analysis-pca-algorithm)
  - [Reconstruction from Compressed Representation](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#reconstruction-from-compressed-representation)
  - [Choosing the number of Principle Components](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#choosing-the-number-of-principle-components)
  - [Advice for Applying PCA](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/2-dimensionality-reduction/#advice-for-applying-pca)
- [Anomaly Detection](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#anomaly-detection)
  - [Problem Motivation](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#problem-motivation)
  - [Applications](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#applications)
  - [The Gaussian Distribution (Normal Distribution)](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#the-gaussian-distribution-normal-distribution)
    - [Computing μ, and σ<sup>2</sup>](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#computing-%CE%BC-and-%CF%83sup2sup)
  - [Algorithm](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#algorithm)
  - [Developing and Evaluating an Anomaly Detection System](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#developing-and-evaluating-an-anomaly-detection-system)
  - [When to use Anomaly Detection vs. Supervised Learning?](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#when-to-use-anomaly-detection-vs-supervised-learning)
  - [Choosing What Features to Use](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#choosing-what-features-to-use)
  - [Multivariate Gaussian Distribution](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#multivariate-gaussian-distribution)
  - [Anomaly Detection using the Multivariate Gaussian Distribution](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#anomaly-detection-using-the-multivariate-gaussian-distribution)
  - [Normal Gaussian Model vs. Multivariate Gaussian Model](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#normal-gaussian-model-vs-multivariate-gaussian-model)
    - [Normal Gaussian Model](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#normal-gaussian-model)
    - [Multivariate Gaussian Model](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/2-unsupervised-learning/3-anomaly-detection/#multivariate-gaussian-model)
