# Support Vector Machines

There's one more algorithm that is very powerful and is very widely used both within industry and academia, and that's called the SVM (support vector machine). SVM sometimes gives a cleaner, and sometimes more powerful way of learning complex non-linear functions. The SVM algorithm is essentially a modified logistic regression algorithm.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Optimization Objective](#optimization-objective)
- [Large Margin Intuition](#large-margin-intuition)
- [Mathematics Behind Large Margin Classification](#mathematics-behind-large-margin-classification)
- [Kernels I](#kernels-i)
- [Kernels II](#kernels-ii)
  - [Choosing the landmarks for Gaussian kernels](#choosing-the-landmarks-for-gaussian-kernels)
  - [Choosing parameters C and σ<sup>2</sup> for Gaussian kernels](#choosing-parameters-c-and-%CF%83sup2sup-for-gaussian-kernels)
- [Using An SVM](#using-an-svm)
  - [Choosing a kernel](#choosing-a-kernel)
    - [Gaussian kernel](#gaussian-kernel)
    - [Linear kernel](#linear-kernel)
    - [Polynomial kernel](#polynomial-kernel)
    - [String kernel](#string-kernel)
  - [Multi-class classification for SVM](#multi-class-classification-for-svm)
  - [When to use Logistic Regression or SVM?](#when-to-use-logistic-regression-or-svm)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Optimization Objective

The SVM provides different way of prioritizing how much we care about optimizing the first term in logistic regression, versus how much we care about optimizing the second term. Unlike logistic regression, the support vector machine doesn't output the probability, it outputs a prediction of y being equal to one or zero, directly.

## Large Margin Intuition

Sometimes people refer to SVM as large margin classifiers. We'll consider what that means and what an SVM hypothesis looks like.

The SVM cost function is as below:

![SVM ](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-.png)

Unlike logistic, <i>h<sub>θ</sub>(x)</i> doesn't give us a probability, but instead we get a direct prediction of 1 or 0
So if <i>θ<sup>T</sup>X</i> is equal to or greater than 0, then <i>h<sub>θ</sub>(x) = 1</i>, otherwise <i>h<sub>θ</sub>(x) = 0</i>. For logistic regression we had two terms;

- Training data set term (i.e. that we sum over m) = A
- Regularization term (i.e. that we sum over n) = B

So we could describe it as A + λB, and it needs some way to deal with the trade-off between regularization and data set terms by setting different values for λ to parametrize this trade-off. Instead of parameterization this as A + λB, for SVMs the convention is to use a different parameter called C, resulting in CA + B, If C were equal to 1/λ then the two functions (CA + B and A + λB) would give the same value.

The SVM cost function is as above, and we've drawn out the cost terms below:

![SVM Cost Terms](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-Cost%20Terms.png)

Left is cost<sub>1</sub> and right is cost<sub>0</sub>. If you have a positive example, you only really need z to be greater or equal to 0, if this is the case then you predict 1. The SVM wants a bit more than that - doesn't want to *just* get it right, but have the value be quite a bit bigger than zero to throw in an extra safety margin factor, logistic regression does something similar.

Consider a case where we set C to be huge number, such as C = 100,000. Considering we're minimizing CA + B, if C is huge we're going to pick an A value so that A is equal to zero, wat is the optimization problem here - how do we make A = 0? Making A = 0, if y = 1, then to make our "A" term 0 we need to find a value of θ so (<i>θ<sup>T</sup>X</i>) is greater than or equal to 1. Similarly, if y = 0, then we want to make "A" = 0 then we need to find a value of θ so (<i>θ<sup>T</sup>X</i>) is equal to or less than -1. If we think of our optimization problem a way to ensure that this first "A" term is equal to 0, we re-factor our optimization problem into just minimizing the "B" (regularization) term, because  when A = 0,  A*C = 0. Turns out when you solve this problem you get interesting decision boundaries:

![SVM Decision Boundaries](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-Decision%20Boundaries.png)

- The green and magenta lines are functional decision boundaries which could be chosen by logistic regression, but they probably don't generalize too well.
- The black line, by contrast is the the chosen by the SVM because of this safety net imposed by the optimization graph. It's a more robust separator. Mathematically, that black line has a larger minimum distance (margin) from any of the training examples.

By separating with the largest margin you incorporate robustness into your decision making process, we looked at this at when C is very large. The SVM is more sophisticated than the large margin might look, but if you were just using large margin then SVM would be very sensitive to outliers because you would risk making a ridiculous hugely impact your classification boundary where a single example might not represent a good reason to change an algorithm. If C is very large then we do use this quite naive maximize the margin approach.

## Mathematics Behind Large Margin Classification

Consider the training example below:

![Mathematics Behind Large Margin Classification Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Mathematics-Behind%20Large%20Margin%20Classification%20Example%20I.png)

Given this data, what boundary will the SVM choose? Note that we're still assuming <i>θ<sub>0</sub> = 0</i>, which means the boundary has to pass through the origin (0,0):

![Mathematics Behind Large Margin Classification Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Mathematics-Behind%20Large%20Margin%20Classification%20Example%20II.png)

SVM would not chose this line because the decision boundary comes very close to examples. **θ is always at 90 degrees to the decision boundary (can show with linear algebra, although we're not going to)**. After projecting a line from <i>x<sup>(1)</sup></i> on to to the θ vector (so it hits at 90 degrees), the distance between the intersection and the origin is (<i>p<sup>1</sup></i>) (the red line). Similarly, the second example (<i>x<sup>(2)</sup></i>) projects a line from <i>x<sup>2</sup></i> into to the θ vector. This is the magenta line, which will be negative (<i>p<sup>2</sup></i>).

We find that both these p values are going to be pretty small. We know we need <i>p<sup>1</sup> \* ||θ||</i> to be bigger than or equal to 1 for positive examples, for this reason, if p is small then that means that ||θ|| must be pretty large. Similarly, for negative examples we need <i>p<sup>2</sup> \* ||θ||</i> to be smaller than or equal to -1. We saw in this example <i>p<sup>2</sup></i> is a small negative number, so ||θ|| must be a large number. Why is this a problem?

The optimization objective is trying to find a set of parameters where the norm of theta is small. So this doesn't seem like a good direction for the parameter vector (because as p values get smaller ||θ|| must get larger to compensate), so we should make p values larger which allows ||θ|| to become smaller.

So lets chose a different boundary:

![Mathematics Behind Large Margin Classification Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Mathematics-Behind%20Large%20Margin%20Classification%20Example%20III.png)

Now if you look at the projection of the examples to θ we find that <i>p<sup>1</sup></i> becomes large and ||θ|| can become small. This means that by choosing this second decision boundary we can make ||θ|| smaller, which is why the SVM choses this hypothesis as better.

## Kernels I

Suppose we have a training set and we want to find a non-linear boundary such as the one in the following image:

![Kernels I Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20I.png)

A way of writing this boundary would be to use a hypothesis that computes a decision boundary by taking the sum of the parameter vector multiplied by a new feature vector f, which simply contains the various high order x terms:

<i>h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub></i>

Where:

- <i>f<sub>1</sub>= x<sub>1</sub></i>
- <i>f<sub>2</sub> = x<sub>1</sub>x<sub>2</sub></i>
- ...

These new features can be defined as similarity functions between x values and landmark values. Have a graph of <i>x<sub>1</sub></i> vs. <i>x<sub>2</sub></i>, then pick three points in that space:

![Kernels I Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20II.png)

These points l<sub>1</sub>, l<sub>2</sub>, and l<sub>3</sub>, which were chosen manually and are called landmarks. Given x, define f<sub>1</sub> as the similarity between (x, l<sub>1</sub>).

The similarity function between x and the landmarks can be defined as:

![Kernels I Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20III.png)

Where:

- `exp` is <i>e</i> to the power of something, e.g. <i>e<sup>z</sup></i>.
- <i>|| x - l<sup>(1)</sup> ||<sup>2</sup></i> is the euclidean distance between the point x and the landmark l<sub>1</sub> squared.
- By statistics, we know that σ is the standard deviation, and σ<sup>2</sup> is commonly called the variance.

So, f2 is defined as:

> <i>f<sub>2</sub> = similarity(x, l<sup>2</sup>) = exp(- (|| x - l<sup>(2)</sup> ||<sup>2</sup> ) / 2σ<sup>2</sup>)</i>

And similarly, f3 is defined as:

> <i>f<sub>3</sub> = similarity(x, l<sup>3</sup>) = exp(- (|| x - l<sup>(3)</sup> ||<sup>2</sup> ) / 2σ<sup>2</sup>)</i>

This similarity function is called a kernel, and this similarity function in particular is a Gaussian kernel. So, instead of writing similarity between x and l we might write <i>f<sub>1</sub> = k(x, l<sub>1</sub></i>).

- Say x is close to a landmark, then the squared distance will be close to zero, or e<sup>-0</sup>, which results in 1.
- Say x is far from a landmark, then the squared distance is big, resulting e to the power of a large number, which is close to zero.

If we plot this behavior, we get a plot like this:

![Kernels I Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20IV.png)

- Notice that when x = [3,5] then f1 = 1.
- As x moves away from [3,5] then the feature takes on values close to zero.
- This measures how close x is to this landmark.

σ<sup>2</sup> is a parameter of the Gaussian kernel and it defines the steepness of the rise around the landmark. We see here that as you move away from [3,5] the feature f1 falls to zero much more slowly if σ<sup>2</sup> is equal to 3.

![Kernels I Example V](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20V.png)

If each feature of our hypothesis is a similarity function between the x values and the landmark values, then we can draw a decision boundary, when for example: <i>θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0</i>

For our example, lets say we've already run an algorithm and got the parameters:

- <i>θ<sub>0</sub> = -0.5</i>
- <i>θ<sub>1</sub> = 1</i>
- <i>θ<sub>2</sub> = 1</i>
- <i>θ<sub>3</sub> = 0</i>

Imagine a point close to f1. We know being close to f1 will result in a value close to 1 because of the Gaussian kernel, but the values of f2 and f3 will be close to 0. So if we look at the formula we have, <i>θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0</i>, -0.5 + 1 + 0 + 0 = 0.5, 0.5 being greater than 0, so we predict 1. If we had another point far away from all three, it would equate to -0.5, so we predict 0.

Considering our parameter, for points near l1 and l2 you predict 1, but for points near l3 you predict 0. Which means we create a non-linear decision boundary that goes like this:

![Kernels I Example VI](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/Kernels-I%20Example%20VI.png)

Where:

- Inside we predict y = 1.
- Outside we predict y = 0.

So this shows how we can create a non-linear boundary with landmarks and the kernel function in the support vector machine.

## Kernels II

How do we get/chose the landmarks? What other kernels can we use (other than the Gaussian kernel)?

### Choosing the landmarks for Gaussian kernels

Given some training data with `m` examples, for each example in the training data, place a landmark at exactly the same location, so we end up with `m` landmarks. If we have one landmark per training example, this means our features measure how close an input would be to a training example.

So given a new example, compute all of the f feature values or f vector, where:

- <i>f<sub>o</sub> = 1</sup></i> always
- <i>f<sub>1</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>1</sup>)</i>
- <i>f<sub>2</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>2</sup>)</i>
- ...
- <i>f<sub>m</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>m</sup>)</i>

When using the Gaussian kernel, somewhere along the line, we will get a f value equal to: <i>f<sub>i</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>i</sup>)</i>, where <i>x<sup>i</sup></i> and <i>l<sup>i</sup></i> would be identical, being equal to e<sup>-0</sup> = 1. We take these `m` f features (fo, f1, f2 ... fm) to compute a [m+1,1] `f` vector.

After obtaining the f vector, the lambda parameters can then be computed using the minimum cost function. Similarly to logistic regression when y = 1 means <i>θ<sup>T</sup>X >= 0</i>, when y = 1 means <i>θ<sup>T</sup>f >= 0</i>. That being said, the minimum cost function of the SVM learning algorithm would be:

![SVM Cost Function](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-Cost%20Function.png)

In this setup m = n, because number of features is the number of training data examples, and now the cost can be computed using f as the feature vector instead of x.

One final mathematic detail, if we ignore <i>θ<sub>0</sub></i> then the following is true:

![SVM Cost Function Theta Term](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-Cost%20Function%20Theta%20Term.png)

But what many implementations do, is instead use this vectorization of theta:

![SVM Cost Function Theta Vectorization](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/9-support-vector-machines/images/SVM-Cost%20Function%20Theta%20Vectorization.png)

Where the matrix M depends on the kernel (e.g. Gaussian kernel). This results in a slightly different minimization or a rescaled version of θ. The reason this is done is because it results in a more efficient computation of the minimum costs for optimum theta parameters, and scale well to much bigger training sets. If you have a training set with 10000 values, then that means you get 10000 features. Solving for all these parameters can become expensive, so by adding this in we avoid a for loop and use a matrix multiplication algorithm instead.

You can apply kernels to other algorithms but they tend to be very computationally expensive such as in linear regression algorithms.

### Choosing parameters C and σ<sup>2</sup> for Gaussian kernels

The parameters C and σ<sup>2</sup> control the bias and variance of the model, this means that varying C and σ<sup>2</sup> results in a trade-off between bias and variance.

**Parameter C:**: C plays a role similar to 1/λ (where λ is the regularization parameter).

- Large C gives a hypothesis of low bias high variance (overfitting).
- Small C gives a hypothesis of high bias low variance (underfitting).

**Parameter σ<sup>2</sup>**: commonly called the variance and it defines the steepness of the rise around the landmark.

- Large σ<sup>2</sup> makes f features vary more smoothly, i.e. higher bias, lower variance.
- Small σ<sup>2</sup> makes f features vary abruptly, i.e. low bias, high variance.

## Using An SVM

It is recommended to use SVM software packages (e.g. liblinear, libsvm) to solve parameters θ, because these packages come with numerical optimizations to calculate optimal θ. However, the parameter C and the kernel function that will be used (e.g. the Gaussian kernel) must be specified.

### Choosing a kernel

Some kernels used in SVMs are:

- Gaussian kernel
- Linear kernel (no kernel)
- Polynomial kernel
- Chi-squared kernel
- Histogram intersection kernel

Linear and Gaussian kernels are the most common. Not all similarity functions are valid kernels because they must satisfy Merecer's Theorem. SVM use numerical optimization tricks, this means certain optimizations can be made, but they must follow the theorem.

Some SVM packages will expect you to define kernel, although, some SVM implementations include the Gaussian and a few others. Gaussian is probably the most popular kernel.

#### Gaussian kernel

- Need to define σ (σ<sup>2</sup>)
- When would you chose a Gaussian kernel?
  - If `n` is small and/or `m` is large, e.g. 2D training set that's large.
- If you're using a Gaussian kernel then you may need to implement the kernel function, e.g. a function <i>f<sub>m</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>m</sup>)</i> which returns a real number.
- Make sure you perform **feature scaling** before using a Gaussian kernel. If you don't, features with a larger values will lead to other features with smaller values to be ignored and made irrelevant to the prediction.

#### Linear kernel

- Predicts y = 1 when <i>θ<sup>T</sup>X >= 0</i>, this means that there is no no f vector, so a standard linear classifier is used.
- Why do this?
  - If n is large and m is small then this means that there are lots of features, but few examples. This means that there is not enough data, so there is a risk of overfitting in a high dimensional feature-space.

Logistic regression and SVM with a linear kernel are pretty similar. They do similar computations and feature similar performance.

#### Polynomial kernel

- Used when x and l are both non-negative
- We measure the similarity of x and l by computing <i>(X<sup>T</sup>L + ε)<sup>d</sup></i>
  - Where:
    - ε is a constant value.
    - d is the deegree of polynomial.
- This means that if X and l are similar, then the inner product tends to be large.
- This is not used that often because other kernels such as the Gaussian kernel tend to be more effective.

#### String kernel

- Used if input is text strings.
- Use for text classification.

### Multi-class classification for SVM

Many packages have built in multi-class classification packages, otherwise the one-vs all method can be used, where a number of K theta parameters are calculated where K is the amount of classes, just like in logistic regression.

### When to use Logistic Regression or SVM?

- If n (features) is large vs. m (training set), then use logistic regression or SVM with a linear kernel, e.g. a text classification problem:
  - Feature vector dimension is 10000 (10000 words)
  - Training set is 10 - 1000.
- If n is small and m is intermediate, then Gaussian kernel is good:
  - n = 1 - 1000.
  - m = 10 - 10000.
- If n is small and m is large, then in that case manually create or add more features, then use logistic regression of SVM with a linear kernel.
  - n = 1 - 1000.
  - m = 50000+.
  - The reason SVM with high computational costs such as Gaussian are not used is because the training set is large, so the model would be too costly too compute and would take a while.

A lot of SVM's power is using diferent kernels to learn complex non-linear functions, for all these regimes a well designed NN should work, but for some of these problems a NN might be slower whereas a well implemented SVM would be faster. SVM also has a convex optimization problem - so you get a global minimum.

SVM is widely perceived as a very powerful learning algorithm. It's not always clear how to chose an algorithm, often more important to get enough data, and designing new features could help or not. The key is training the different models and perform the error analysis with a cross-validation data set for multiple models, then choose the one with the lowest error on the test data set.
