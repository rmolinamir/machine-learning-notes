# Classification

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x<sup>(i)</sup> may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y ∈ {0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given x<sup>(i)</sup>, the corresponding y<sup>(i)</sup> is also called the label for the training example.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Hypothesis Representation](#hypothesis-representation)
- [Logistic Regression Cost Function](#logistic-regression-cost-function)
- [Advanced Optimization Algorithms for Logistic Regresion](#advanced-optimization-algorithms-for-logistic-regresion)
- [Multiclass Classification: One-vs-all](#multiclass-classification-one-vs-all)
- [Overfitting](#overfitting)
- [Regularization Cost Function](#regularization-cost-function)
- [Regularized Linear Regression](#regularized-linear-regression)
  - [Regularized Linear Regression Gradient Descent](#regularized-linear-regression-gradient-descent)
  - [Regularized Linear Regression Normal Equation](#regularized-linear-regression-normal-equation)
- [Regularized Logistic Regression](#regularized-logistic-regression)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Hypothesis Representation

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense <i>h<sub>θ</sub>(x)</i> to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses <i>h<sub>θ</sub>(x)</i> to satisfy:

> 0 <= <i>h<sub>θ</sub>(x)</i> <= 1

This is accomplished by plugging <i>θ<sup>T</sup>x</i> into the Logistic Function. Our new form uses the "Sigmoid Function," also called the "Logistic Function":

![Sigmoid Function](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Sigmoid-Function.png)

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

<i>h<sub>θ</sub>(x)</i>will give us the probability that our output is 1. For example, <i>h<sub>θ</sub>(x)</i> = 0.7</i> gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

## Logistic Regression Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function. Instead, the cost function will vary depending if the value of y is equal to 1 or 0.

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

Notice that the algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

![Cost Function Vectorized Implementation](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Cost-Function%20Vectorized%20Implementation.png)

## Advanced Optimization Algorithms for Logistic Regresion

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. Do not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized.

## Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}. Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

## Overfitting

Overfitting refers to hypothesis functions that are "overly" fitted to the traning set. These hypothesis functions may have a cost function value very close to zero or zero, but they are generally not accurate when predicting values outside the training zet. In contrast, underfitting hypothesis functions do not capture the data very well.

Underfitting, or high bias, is when the form of our hypothesis function maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:
      - Manually select which features to keep.
      - Use a model selection algorithm (studied later in the course).
2) Regularization:
      - Keep all the features, but reduce the magnitude of parameters <i>θ<sub>j</sub></i>.
      - Regularization works well when we have a lot of slightly useful features.

## Regularization Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

The λ, or lambda, is the regularization parameter and it regulates the theta parameters. It determines how much the costs of our theta parameters are inflated.

Using the cost function with the extra summation of the regularization parameter keeping the theta parameters in check, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if lambda is 0 or is too small? Nothing, but no regularization would be in place.

## Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression.

### Regularized Linear Regression Gradient Descent

We will modify our gradient descent function to separate out θ<sub>θ</sub> from the rest of the parameters because we do not want to penalize θ<sub>0</sub>.

![Regularized Linear Regression Gradient Descent Equation](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Regularized-Linear%20Regression%20Gradient%20Descent%20Equation.png)

The term accompanying θ<sub>j</sub> performs our regularization. With some manipulation our update rule can also be represented as:

![Regularized Linear Regression Gradient Descent Equation 2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Regularized-Linear%20Regression%20Gradient%20Descent%20Equation%202.png)

The first term in the above equation accompanying lambda, will always be less than 1. Intuitively you can see it as reducing the value of theta by some amount on every update. Notice that the second term is now exactly the same as it was before.

### Regularized Linear Regression Normal Equation

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

![Regularized Linear Regression Normal Equation](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Regularized-Linear%20Regression%20Normal%20Equation.png)

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including <i>x<sub>0</sub></i>), multiplied with a single real number λ.

## Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![Regularized logistic Regression Normal Equation](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Regularized-logistic%20Regression%20Normal%20Equation.png)

![Regularized logistic Regression Normal Equation 2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/4-classification/images/Regularized-logistic%20Regression%20Normal%20Equation%202.png)
