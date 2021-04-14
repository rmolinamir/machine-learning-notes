# Diagnostics

Machine Learning Diagnostics is a test you can run, to get insight into what is or isn't working with an algorithm, and which will often give you insight as to what are promising things to try to improve a learning algorithm's performance.

Suppose that after you take your learnt parameters, if you test your hypothesis on the new set of data, suppose you find that this is making huge errors in the predictions. The question is what should you then try mixing in order to improve the learning algorithm? There are many things that one can think of that could improve the performance of the learning algorithm, such as:

- **Getting more training examples:**  Fixes high variance
- **Trying smaller sets of features:** Fixes high variance
- **Trying additional features:** Fixes high bias
- **Trying polynomial features:** Fixes high bias
- **Decreasing λ:** Fixes high bias
- **Increasing λ:** Fixes high variance

Fortunately, there is a pretty simple technique that can let you very quickly rule out half of the things on this list as being potentially promising things to pursue. And there is a very simple technique, that if you run, can easily rule out many of these options, and potentially save you a lot of time pursuing something that's just is not going to work.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Diagnosing Neural Networks](#diagnosing-neural-networks)
- [Evaluating a Hypothesis](#evaluating-a-hypothesis)
- [Model Selection and Train/Validation/Test Sets](#model-selection-and-trainvalidationtest-sets)
- [Diagnosing Bias vs. Variance](#diagnosing-bias-vs-variance)
- [Regularization and Bias/Variance](#regularization-and-biasvariance)
- [Learning Curves](#learning-curves)
- [Neural Networks and Overfitting](#neural-networks-and-overfitting)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Diagnosing Neural Networks

A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper. A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.
Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

Model Complexity Effects:

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
- In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

## Evaluating a Hypothesis

Once we have done some trouble shooting for errors in our predictions, we can move on to evaluate our new hypothesis. A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a training set and a test set. Typically, the training set consists of 70% of your data and the test set is the remaining 30%.

The new procedure using these two sets is then:

1. Learn Θ and minimize <i>J<sub>train</sub>(Θ)</i> using the training set
2. Compute the test set error <i>J<sub>test</sub>(Θ)</i>

To computer the test set error:

![The test set error](https://i.imgur.com/qBLc8s1.png)

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

![Test Error](blob:https://imgur.com/257d6e08-f397-43a6-9f21-925b6a09eef7)

This gives us the proportion of the test data that was misclassified.

## Model Selection and Train/Validation/Test Sets

Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to break down our dataset into the three sets is:

- Training set: 60%
- Cross validation set: 20%
- Test set: 20%

We can now calculate three separate error values for the three different sets using the following method:

1. Optimize the parameters in Θ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with <i>J<sub>test</sub>((Θ)<sup>(d)</sup>)</i>, (d = theta from polynomial with lower error);

This way, the degree of the polynomial d has not been trained using the test set.

## Diagnosing Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

We need to distinguish whether bias or variance is the problem contributing to bad predictions.
High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.
The training error will tend to decrease as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to decrease as we increase d up to a point, and then it will increase as d is increased, forming a convex curve.

High bias (underfitting): both the training and the cross validation errors will be high. Also, the training error will be approximately equal to the cross validation error.

High variance (overfitting): the training error is low, but the cross validation error will be much greater than the training error.

The is summarized in the figure below:

![Diagnosing Bias vs. Variance](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/I4dRkz_pEeeHpAqQsW8qwg_bed7efdd48c13e8f75624c817fb39684_fixed.png?expiry=1617926400000&hmac=VZoumHH6rUt2FxeDXqVxBXWUH3FjZITom9G117kPRZ0)

## Regularization and Bias/Variance

How do we choose our parameter λ to get it 'just right'? In order to choose the model and the regularization term λ, we need to:

1. Create a list of lambdas (i.e. λ ∈ {0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
2. Create a set of models with different degrees or any other variants.
3. Iterate through the λs and for each λ go through all the models to learn some Θ.
4. Compute the cross validation error using the learned Θ (computed with λ) on the cross validation cost function **without** regularization or λ = 0.
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo Θ and λ, apply it on the test cost function and calculate the error to see if it has a good generalization of the problem.

![Regularization and Bias/Variance](https://i.imgur.com/TKIIMuz.png)

## Learning Curves

Training an algorithm on a very few number of data points (such as 1, 2 or 3) will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. Hence:

- As the training set gets larger, the error for a quadratic function increases.
- The error value will plateau out after a certain m, or training set size.

**Experiencing high bias:**

**Low training set size:** causes the training cost to be low and the test cost to be high.

**Large training set size:** causes both the training cost and the test cost to be high and approximately equal values.

**If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.**

![Typical Learning Curve for High Bias](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bpAOvt9uEeaQlg5FcsXQDA_ecad653e01ee824b231ff8b5df7208d9_2-am.png?expiry=1618012800000&hmac=HqstDFHVhZxUr92hEutpTCX4rSVV1rYKRJm1FNPwq7g)

**Experiencing high variance:**

**Low training set size:** the training cost will be low and the test cost will be high.

**Large training set size:** the training cost increases with training set size and the test cost continues to decrease without leveling off. Also, the training cost is lesser than the test cost but the difference between them remains significant.

**If a learning algorithm is suffering from high variance, getting more training data is likely to help.**

![Typical Learning Curve for High Variance](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/vqlG7t9uEeaizBK307J26A_3e3e9f42b5e3ce9e3466a0416c4368ee_ITu3antfEeam4BLcQYZr8Q_37fe6be97e7b0740d1871ba99d4c2ed9_300px-Learning1.png?expiry=1618012800000&hmac=_FIJ-PUK9rv3oiB91WgWdapRF0p6BauQaTCsHDmM4iY)

## Neural Networks and Overfitting

- Small neural networks (fewer parameters) are more prone to underfitting, but it is computationally cheaper.
- Large neural networks (more parameters) are more prone to overfitting, but it is computationally more expensive. Use regularization to address overfitting.
