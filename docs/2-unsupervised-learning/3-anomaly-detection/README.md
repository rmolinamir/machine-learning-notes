# Anomaly Detection

Anomaly detection is widely used in monitoring operations such as fraud detection (e.g. ‘has this credit card been stolen?’). Given a large number of data points, we may sometimes want to figure out which ones vary significantly from the average. For example, in manufacturing, we may want to detect defects or anomalies. We show how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Anomaly Detection](#anomaly-detection)
  - [Problem Motivation](#problem-motivation)
  - [Applications](#applications)
  - [The Gaussian Distribution (Normal Distribution)](#the-gaussian-distribution-normal-distribution)
    - [Computing μ, and σ<sup>2</sup>](#computing-%CE%BC-and-%CF%83sup2sup)
  - [Algorithm](#algorithm)
  - [Developing and Evaluating an Anomaly Detection System](#developing-and-evaluating-an-anomaly-detection-system)
  - [When to use Anomaly Detection vs. Supervised Learning?](#when-to-use-anomaly-detection-vs-supervised-learning)
  - [Choosing What Features to Use](#choosing-what-features-to-use)
  - [Multivariate Gaussian Distribution](#multivariate-gaussian-distribution)
  - [Anomaly Detection using the Multivariate Gaussian Distribution](#anomaly-detection-using-the-multivariate-gaussian-distribution)
  - [Normal Gaussian Model vs. Multivariate Gaussian Model](#normal-gaussian-model-vs-multivariate-gaussian-model)
    - [Normal Gaussian Model](#normal-gaussian-model)
    - [Multivariate Gaussian Model](#multivariate-gaussian-model)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem Motivation

Anomaly detection is a reasonably commonly used type of machine learning application, it can be thought of as a solution to an unsupervised learning problem but it also has aspects of supervised learning. What is anomaly detection? Imagine you're an aircraft engine manufacturer, as engines roll off your assembly line you're doing QA to measure certain features from engines (e.g. vibration and heat generated of/by the engines), you now have a dataset of x<sup>1</sup> to x<sup>m</sup> examples, i.e. <i>m</i> engines were tested. Say we plot that dataset:

![Problem Motivation Example I](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image.png)

Then, a new engine is measured. An anomaly detection method is used to see if the new engine is anomalous compared to the previous engines. When plotted it looks *probably* fine, the new engine looks like this:

![Problem Motivation Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[1].png)

Finally, we measure a new engine that looks like this, which looks like an **anomalous data-point**:

![Problem Motivation Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[2].png)

More formally, we have a dataset which contains normally distributed data. It's up to us how to ensure they're normally distributed, and using that dataset as a reference point we can see whether other examples are anomalous. In reality, if the data is not normally distributed then it should be fine as well.

To do this, first, using our training dataset we build a model for p of x, which will be like asking, "What is the probability that example <i>x</i> is normal?". In other words, we're going to build a model for the probability of x (<i>p(x)</i>), where for example <i>x</i> are features of aircraft engines. And so, having built a model of the probability of x we're then going to say that for the new aircraft engine (x-test), if p of x-test is less than an *epsilon* value then we flag this as an anomaly.

In summary:

- If <i>p(x<sub>test</sub>) < ε</i>: An anomaly.
- If <i>p(x<sub>test</sub>) >= ε</i>: Not an anomaly.

The parameter ε is a threshold for the probability value which determines whether or not the data point is an anomaly, and it is defined by us depending on how strict we want the model to be. We expect our model to (graphically) look something like this:

![Problem Motivation Example IV](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[3].png)

## Applications

Here are some examples of applications of anomaly detection:

**Fraud detection:** Perhaps the most common application of anomaly detection is actually for detection. Users have activity associated with them, such as length on time online, location of login, spending frequency, etc. Using this data we can build a model of what normal users' activity is like, then ask "what is the probability of normal behavior?". We can identify unusual users by sending their data through the model, flag up anything that looks a bit weird, automatically block cards/transactions, et al.

**Manufacturing:** Where you can find unusual manufactured products, for example, anomalous aircraft engines and send those for further review.

**Monitoring computers in a data center:** A third application would be monitoring computers in a data center. If you have a lot of machines in a computer cluster or in a data center, we can do things like compute features at each machine, features such as, how much memory is used, number of disc accesses, CPU load, etc. As well as more complex features like what is the CPU load on this machine divided by the amount of network traffic on this machine. Then given the dataset of how your computers in your data center usually behave, you can model the probability of x, so you can model the probability of these machines having different amounts of memory use or probability of these machines having different numbers of disc accesses or different CPU loads, and so on. This is actually being used today by various data centers to watch out for unusual things happening on their machines so that they can be flagged for review by a system administrator.

## The Gaussian Distribution (Normal Distribution)

Say we have a data set <i>x</i> that contains real numbers R, where:

- The mean will be defined by μ.
- The variance will be defined by σ<sup>2</sup>.
- σ is also called the **standard deviation** - which affects the width of the Gaussian curve.

Then we can write ~N(μ, σ<sup>2</sup>), where:

- ~ indicates "is distributed as".
- N indicates "a normal distribution".
- μ, σ<sup>2</sup> represent the mean and variance, respectively.

The distribution looks like this:

![The Gaussian Distribution (Normal Distribution) Example I](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[4].png)

This curve specifies the probability of <i>x</i> taking on different values as you move away from μ, i.e. x taking on values in the middle is high, since the Gaussian density is high, whereas x taking on values further, and further away will be diminishing in probability.

The Gaussian equation is <i>P(x : μ, σ<sup>2</sup>)</i>, which is read as "probability of x, parameterized by the mean and squared variance", and can be written as:

![The Gaussian Distribution (Normal Distribution) Equation](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[5].png)

Some examples of Gaussians as follows, where the area under the curve must integrate to one as this is a property of probability distributing (i.e. the area under the curve is always equal to 1), so the width and height of the curve changes as the standard deviation changes, while the area remains the same.

![The Gaussian Distribution (Normal Distribution) Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[6].png)

### Computing μ, and σ<sup>2</sup>

We can do the following to compute the parameters μ (mean) and σ<sup>2</sup> (variance). Say we have a data set of <i>m</i> examples. Give each example is a real number - we can plot the data on the horizontal axis as shown below:

![The Gaussian Distribution (Normal Distribution) Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[7].png)

Supposing these examples follow a Gaussian distribution, given the dataset, can you estimate the distribution? A reasonable fit for the data would look like the image below, where there is a higher probability of values being in the central region, and a lower probability of being further away:

![The Gaussian Distribution (Normal Distribution) Example IV](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[8].png)

To compute μ and σ<sup>2</sup>:

![The Gaussian Distribution (Normal Distribution) Mean and Variance Equation](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[9].png)

These parameters are the maximum likelihood estimation values for μ and σ<sup>2</sup>.

## Algorithm

Given an unlabeled training set of m examples where each example is an n-dimensional vector (i.e. a feature vector containing <i>n</i> features), we can model <i>p(x)</i> from the data set knowing which are the high probability and low probability features, with x as a vector, as:

p(x) = p(x<sub>1</sub>; μ<sub>1</sub>, σ<sub>1</sub><sup>2</sup>) * p(x<sub>2</sub>; μ<sub>2</sub>, σ<sub>2</sub><sup>2</sup>) ... p(x<sub>n</sub>; μ<sub>n</sub>, σ<sub>n</sub><sup>2</sup>)

The model is the product between the model of all features, meaning, the model of each of the features are computed assuming they are normally distributed, where:

p(x<sub>i</sub>; μ<sub>i</sub>, σ<sub>i</sub><sup>2</sup>) is the probability of feature x<sub>i</sub> given μ<sub>i</sub> and σ<sub>i</sub><sup>2</sup>, assuming a normal distribution.

> This equation makes an **independence assumption** for the features, although the algorithm works if the features are independent or not. Don't worry too much about this, although, if the features are tightly linked then you should be able to do some dimensionality reduction anyway!

We can write this chain of multiplication more compactly as follows:

![Algorithm Model](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[10].png)

Where Π is the product of a set of values.

With these concepts in mind, the algorithm can be executed as follows:

![Algorithm Implementation](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[11].png)

1. Chose features:
   - Try to come up with features which might help identify something anomalous - may be unusually large or small values.
   - More generally, chose features which describe the general properties.
   - This is nothing unique to anomaly detection - it's just the idea of building a sensible feature vector.
2. Fit parameters:
   - Determine parameters for each of your examples μ<sub>i</sub> and σ<sub>i</sub><sup>2</sup>.
   - Compute the standard deviation and the mean for each feature.
   - Use a vectorized implementation rather than a loop.
3. Compute p(x):
   - Compute the formula shown (the formula for the Gaussian probability).
   - If the number is very small, it means that there is a very low chance of it being "normal".

For example:

Given features x<sub>1</sub> and x<sub>2</sub> where:

![Algorithm Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[12].png)

- x<sub>1</sub>:
  - Mean is about 5.
  - Standard deviation looks to be about 2.
- x<sub>2</sub>:
  - Mean is about 3.
  - Standard deviation about 1.

If we plot the Gaussian for x<sub>1</sub> and x<sub>2</sub> we get a figure like this:

![Algorithm Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[13].png)

If you plot the product of these models, you get a surface plot like this:

![Algorithm Example IV](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[14].png)

- With this surface plot, the height of the surface is the probability - p(x).
- We can't always do surface plots, but for this example it's quite a nice way to show the probability of a 2D feature vector.

If you look at the surface plot, all values above a certain height (a threshold) are flagged as normal, all the values below that threshold are probably anomalous.

To check if a value is anomalous:

- Set epsilon as some value
- Say we have two new data points:
  - x<sup>1</sup><sub>test</sub>
  - x<sup>2</sup><sub>test</sub>

We compute:

- p(x<sup>1</sup><sub>test</sub>) = 0.436 >= epsilon (~40% chance it's normal and above the threshold, therefore marked as normal)
- p(x<sup>2</sup><sub>test</sub>) = 0.0021 < epsilon (~0.2% chance it's normal and below the threshold, therefore marked as anomalous)

## Developing and Evaluating an Anomaly Detection System

How can we develop a system for anomaly detection and evaluate an algorithm? Previously we spoke about the importance of real-number evaluation (e.g. error analysis and F1-score).

Often, we need to make a lot of choices (e.g. features to use). It's easier to evaluate your algorithm if it returns a single number to show if changes you made improved or worsened an algorithm's performance.

To develop an anomaly detection system quickly, it would be helpful to have a way to evaluate your algorithm.

Assume we have some labeled data. So far we've been treating anomalous detection with unlabeled data. labeled data allows evaluation, i.e. if you think something is anomalous you can be sure if it is or not.

Say, you have some labeled data:

- Data for engines which were non-anomalous (y = 0).
- Data for engines which were anomalous (y = 1).

Training set is the collection of normal examples, and it's fine even if we have only a few anomalous data examples.

Next we define the cross validation set, and the test set. For both assume we can include a few examples which have anomalous examples.

In our engine example:

- We have 10000 good engines (it's fine even if a few bad ones are here). This means we have many examples with y = 0.
- We have 20 flawed engines (typically when y = 1).

We can split this data set into:

- Training set: 6000 good engines (y = 0).
- CV set: 2000 good engines, 10 anomalous.
- Test set: 2000 good engines, 10 anomalous.
- Ratio is 3:1:1.

To evaluate the algorithm:

- Take trainings set {x<sup>1</sup>, x<sup>2</sup>, ..., x<sup>m</sup>} and compute model p(x).
- On cross validation and test set, test the example x:
  - y = 1 if p(x) < epsilon (anomalous)
  - y = 0 if p(x) >= epsilon (normal)
- Think of the algorithm as trying to predict if something is anomalous.
- Since we have labeled data, we can check if it's working correctly.
- It look like a supervised learning algorithm.

`y = 0` is good metric to use for evaluation, but it is very common in comparison to `y = 1`, i.e. the data is skewed, so classification would not be ideal. Therefore, we perform an error analysis:

- Compute a fraction of true positives/false positives/false negatives/true negatives.
- Compute the precision/recall.
- Compute the F1-score.

The threshold value epsilon is used to determine when something is anomalous.

With the cross validation set we can see how varying epsilon effects various evaluation metrics, then pick the value of epsilon which maximizes the score on the cross validation set. So we can:

- Evaluate the algorithm using cross validation set
- Do a final algorithm evaluation on the test set.

## When to use Anomaly Detection vs. Supervised Learning?

If we have labeled data, why not use a supervised learning algorithm? Here we'll try and understand when you should use supervised learning and when anomaly detection would be better.

**Anomaly Detection:**

- Very small number of positive examples:
  - Save positive examples just for CV and test set.
  - Consider using an anomaly detection algorithm.
  - Not enough data to "learn" positive examples.
- Skewed data, i.e. a very large number of negative examples relative to positive examples.
  - Use these negative examples to compute p(x).
  - Positive examples are not needed to compute p(x).
- Many "types" of anomalies:
  - Hard for an algorithm to learn from positive examples when anomalies may look nothing like one another:
    - So anomaly detection doesn't know what they look like, but knows what they don't look like.
  - When we looked at spam email:
    - Many types of spam.
    - For the spam problem, there are usually enough positive examples. This is why we usually think of spam as supervised learning.
- Application and why they're anomaly detection:
  - Fraud detection:
    - Many ways you may do fraud.
    - If you're a major on line retailer/very subject to attacks, sometimes might shift to supervised learning.
  - Manufacturing:
    - If you make huge volumes we likely have enough positive data, so a supervised learning model can also be done.
      - This means you make an assumption about the kinds of errors you're going to see.
      - It's the unknown unknowns we don't like.

**Supervised learning:**

- Reasonably large number of positive and negative examples.
- We have enough positive examples to give an algorithm the opportunity to learn what they look like.
  - If you expect anomalies to look anomalous in the same way.
- Applications:
  - Email/spam classification.
  - Weather prediction.
  - Cancer classification.

## Choosing What Features to Use

Deciding which features to use has a huge impact on the prediction model.

One thing that it's often done would be to plot the data or the histogram of the data, to make sure that the data looks vaguely Gaussian before feeding it to my anomaly detection algorithm.

Plotting a histogram of the data to check it has a Gaussian distribution is a nice sanity check, but even if the feature looks **non-Gaussian** then often the data still works.

Non-Gaussian data might look like this:

![Choosing What Features to Use Example I](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[15].png)

We can transform the data to make it look more like a Gaussian distribution so it's also very useful to also visualize the data. We can play with different transformations of the data to make it look more Gaussian, i.e. if you have some feature x<sub>1</sub>, replace it with log(x<sub>1</sub>), log(x<sub>1</sub> + c), or with x<sub>1</sub><sup>1/2</sup>, x<sub>1</sub><sup>1/3</sup>. After transforming the data, it might look like this:

![Choosing What Features to Use Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[16].png)

If you plot a histogram with the data, and find that it looks pretty non-Gaussian, it's worth playing around a little bit with different transformations like these, to see if you can make your data look a little bit more Gaussian, before you feed it to your learning algorithm, although even if you don't, it might work okay.

But, how do you come up with features for an anomaly detection algorithm? Via an error analysis procedure.

Just like in a supervised learning error analysis procedure we:

- Run the algorithm on the CV set.
- See which examples it got wrong.
- Develop new features based on trying to understand *why* the algorithm got those examples wrong.

For example, in the picture below we have computed the normal distribution of one feature, and our anomalous example (green cross) is in a place where usually you'd expect values to be non-anomalous (e.g. close to the mean):

![Choosing What Features to Use Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[17].png)

To deal we this, we can:

- Look at the data - see what went wrong.
- Develop a new feature (x<sub>2</sub>) which can help distinguish the anomaly.

For example, in a data center we could have features such as:

- x<sub>1</sub> = Memory use.
- x<sub>2</sub> = Number of disk access/sec.
- x<sub>3</sub> = CPU load.
- x<sub>4</sub> = Network Traffic.

We suspect that the CPU load and network traffic grow linearly with one another. If the server is serving many users, CPU is high and network is high, but if we have an example where the CPU load grows but network traffic is low, then that is indicative of an anomaly.

So we implement a new feature that represents CPU load/network traffic (CPU load per network traffic), we may also need to do feature scaling.

## Multivariate Gaussian Distribution

This is a slightly different technique which can sometimes catch some anomalies in cases where non-multivariate Gaussian distribution anomaly detection systems fails to. An unlabeled data set looks like this:

![Multivariate Gaussian Distribution Example I](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[18].png)

Say you can fit a Gaussian distribution to CPU load and memory use, in the test set we have an example which looks like an anomaly (e.g. x<sub>1</sub> = 0.4, x<sub>2</sub> = 1.5). As shown in the picture below, this example (green cross) has high memory and low CPU load (if we plot x<sub>1</sub> vs. x<sub>2</sub>, this example is far away from the others).

![Multivariate Gaussian Distribution Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[19].png)

Problem is, if we look at each feature individually they fall within acceptable limits - but if we analyse both features of this example collectively, we realize we shouldn't be getting those kind of values even though or prediction model is not flaggin this example as anomalous.

This is because our function computes the probability prediction in concentric circles (like a contour plot), it thinks that everything in each of the circular regions have about an equal probability, and it doesn't realize that the example (green cross) actually has much lower probability than the others.

![Multivariate Gaussian Distribution Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[20].png)

To address this we develop the multivariate Gaussian distribution where we model p(x) all in one go, instead of each feature separately.

The parameters for this new model are:

- μ - which is an n dimensional vector (where n is number of features).
- Σ - which is an [n x n] matrix - the covariance matrix.

The formula for the multivariate Gaussian distribution is as follows:

![Multivariate Gaussian Distribution Example IV](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[21].png)

Where:

- |Σ| is the absolute value of Σ (determinant of sigma).

More importantly, what do the parameters of our prediction model p(x) look like? In a 2D example, our parameters could look like this:

![Multivariate Gaussian Distribution Example VI](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[23].png)

Where Sigma is sometimes call the identity matrix. After computing p(x), our model would look like this:

![Multivariate Gaussian Distribution Example VII](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[24].png)

Where the height of the surface indicates the value of p(x) and it's determined by the features x<sub>1</sub> and x<sub>2</sub>.

What happens to our probability model if we change Sigma (e.g. the identify matrix) to look like this?

![Multivariate Gaussian Distribution Example VIII](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[25].png)

![Multivariate Gaussian Distribution Example IX](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[26].png)

What you get is that the width of the bump diminishes and its height increases. So if we set Sigma to be different values, we change the shape of our graph.

![Multivariate Gaussian Distribution Example X](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[27].png)

Using these values we can, therefore, define the shape of this to better fit the data, rather than assuming symmetry in every dimension. One of the cool things is you can use it to model correlation between data, if you start to change the off-diagonal values in the covariance matrix you can control how well the various dimensions correlation:

![Multivariate Gaussian Distribution Example XI](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[28].png)

- In the final example, we can see that it computes a very tall thin distribution, i.e. it shows a strong positive correlation.
- We can also make the off-diagonal values negative to produce a negative correlation.
- We can also move the mean (μ) which varies the peak of the distribution.

This shows an example of the kinds of distribution you can get by varying the parameters of our model.

## Anomaly Detection using the Multivariate Gaussian Distribution

Previously, we saw some examples of the kinds of distributions you can model, now let's take those ideas and look at applying them to different anomaly detection algorithms. As mentioned, multivariate Gaussian modeling uses the following equation:

![Anomaly Detection using the Multivariate Gaussian Distribution Example I](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[29].png)

Which comes with the parameters μ and Σ, where:

- μ is the mean (n-dimenisonal vector).
- Σ is the covariance matrix ([n x n] matrix).

To fit our parameters, we can estimate their optimum values depending on our examples. If we have a set of examples: {x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ...x<sub>n</sub>}, then the formulas for estimating our μ and Σ parameters are:

![Anomaly Detection using the Multivariate Gaussian Distribution Example II](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[30].png)
![Anomaly Detection using the Multivariate Gaussian Distribution Example III](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[31].png)

With this formulas in mind, the algorithm to compute a multivariate Gaussian distribution model would follow this process:

1. Calculate μ and Σ using the formula above with the training data set.
2. Given a new example (x<sub>test</sub>, represented by the green cross) - see below, compute p(x) using the following formula for multivariate distribution:
    ![Anomaly Detection using the Multivariate Gaussian Distribution Example IV](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[33].png)
    ![Anomaly Detection using the Multivariate Gaussian Distribution Example V](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[32].png)
3. Compare the value with the threshold probability value, ε:
   - If <i>p(x<sub>test</sub>) < ε</i>: An anomaly.
   - If <i>p(x<sub>test</sub>) >= ε</i>: Not an anomaly.

Our model would end up looking similar to this:

![Anomaly Detection using the Multivariate Gaussian Distribution Example VI](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[34].png)

Which means it's likely to identify the green value as anomalous. Finally, we should mention how multivariate Gaussian relates to our original simple Gaussian model (where each feature is looked at individually).

The original model corresponds to a multivariate Gaussian where the Gaussians' contours are axis aligned (i.e. the parameters μ and Σ are not modified, so our contours are perfectly circular), this means that the normal Gaussian model is a special case of multivariate Gaussian distribution.

This can be shown mathematically with the constraint that the covariance matrix sigma has zero values on the non-diagonal values:

![Anomaly Detection using the Multivariate Gaussian Distribution Example VII](https://www.holehouse.org/mlclass/15_Anomaly_Detection_files/Image%20[35].png)

If you compute the multivariate model with these parameters, it would end up being identical to the normal Gaussian model.

## Normal Gaussian Model vs. Multivariate Gaussian Model

### Normal Gaussian Model

- Probably used more often.
- it might be necessary to manually create features to capture anomalies where x<sub>1</sub> and x<sub>2</sub> take unusual combinations of values.
  - It needs to make extra features.
  - Might not be obvious what they should be.
    - This is always a risk - where you're using your own expectation of a problem to "predict" future anomalies.
    - Typically, anomalies that are caught initially aren't going to be anomalies caught sometime in the future, if you thought of them they'd probably be avoided in the first place.
      - This is a bigger issue, and one which may or may not be relevant depending on your problem space.
- Much cheaper computationally.
- Scales much better to very large feature vectors (because we don't have to compute the inverse of the covariance matrix Σ with dimensions [n x n]).
- Even if n = 100000 the normal model works fine, it works well with small training sets too (e.g. 50 - 100).
- Because of these factors it's used more often. It represents an optimized but axis-symmetric specialization of the general model.

### Multivariate Gaussian Model

- Used less frequently.
- Can "automatically" capture feature correlation, so there is no need to create additional features.
- Less computationally efficient, i.e. more costly to compute.
  - Must compute inverse of Σ which is [n x n].
- As a property of the algorithm, it needs the number of examples must be greater than the number of features(m > n).
  - If this is not true, this results in Σ being a singular matrix (non-invertible).
  - In reality, the number of examples should be much greater than the number of features (m >>> n).
- If you find the matrix is non-invertible, it could be for one of two main reasons:
  - m < n, so use original simple model.
  - Redundant features (e.g. linearly dependent, two features that are the same).
    - If this is the case, you could use PCA or sanity check your data set.
