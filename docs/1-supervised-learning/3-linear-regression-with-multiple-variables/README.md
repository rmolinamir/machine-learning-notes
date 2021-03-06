# Linear Regression with Multiple Variables (or Features)

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

> - x<sub>j</sub><sup>(i)</sup> = value of feature j in the ith training example
> - x<sup>(i)</sup> = the input (features) of the ith training example
> - m = the number of training examples
> - n = the number of features

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

> h<sub>θ</sub>(x) = θ<sup>T</sup>X

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![Linear Regression with Multiple Variables Equation Comparison](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Linear-Regression%20with%20Multiple%20Variables%20Equation%20Comparison.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Gradient Descent in Practice I: Feature Scaling](#gradient-descent-in-practice-i-feature-scaling)
- [Gradient Descent in Practice I: Learning Rate](#gradient-descent-in-practice-i-learning-rate)
- [Features and Polynomial Regression](#features-and-polynomial-regression)
- [Normal Equation](#normal-equation)
- [Normal Equation Noninvertibility](#normal-equation-noninvertibility)
- [Vectorized Implementations](#vectorized-implementations)
  - [Sources](#sources)
  - [Hypothesis Vectorization](#hypothesis-vectorization)
  - [Cost Function Vectorization](#cost-function-vectorization)
  - [Cost Function Derivation Vectorization](#cost-function-derivation-vectorization)
  - ["Side" Mentions](#side-mentions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Gradient Descent in Practice I: Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

![Feature Scaling Range 1](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Feature-Scaling%20Range%201.png)

![Feature Scaling Range 2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Feature-Scaling%20Range%202.png)

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are feature scaling and mean normalization. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

![Feature Scaling Equation](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Feature-Scaling%20Equation.png)

Where <i>μ<sub>i</sub></i> is the average of all the values for feature (i) and <i>s<sub>i</sub></i> is the range of values (max - min), or <i>s<sub>i</sub></i> is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if Xi represents housing prices with a range of 100 to 2000  and a mean value of 1000, then:

![Feature Scaling Equation Example](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Feature-Scaling%20Equation%20Example.png)

## Gradient Descent in Practice I: Learning Rate

**Debugging gradient descent:** Make a plot with number of iterations on the x-axis. Now plot the cost function J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

**Automatic convergence test:** Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10<sup>-3</sup>. However in practice it's difficult to choose this threshold value. It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.

To summarize:

- If α is too small: slow convergence.
- If α is too large: may not decrease on every iteration and thus may not converge.

## Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine x<sub>1</sub> and x<sub>2</sub> into a new feature x<sub>3</sub> by taking x<sub>1</sub>⋅ x<sub>2</sub>.

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> then we can create additional features based on x<sub>1</sub>, to get the quadratic function h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>1</sub><sup>2</sup>  or the cubic function h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>1</sub><sup>2</sup> + θ<sub>3</sub>x<sub>1</sub><sup>3</sup>.

In the cubic version, we have created new features x<sub>2</sub> and x<sub>3</sub>, where x<sub>2</sub> = x<sub>1</sub><sup>2</sup> and x<sub>3</sub> = x<sub>1</sub><sup>3</sup>.

To make it a square root function, we could do:

![Features and Polynomial Regression Square Root Function](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Features-and%20Polynomial%20Regression%20Square%20Root%20Function.png)

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important. For example, if x<sub>1</sub> has range 1 - 1000 then range of x<sub>1</sub><sup>2</sup> becomes 1 - 1000000 and that of x<sub>1</sub><sup>3</sup> becomes 1 - 1000000000.

## Normal Equation

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:

<i>θ = (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> Y</i>

There is no need to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent           | Normal Equation                                 |
|----------------------------|-------------------------------------------------|
| Need to choose alpha       | No need to choose alpha                         |
| Needs many iterations      | No need to iterate                              |
| O (kn^2kn2)                | O (n^3n3), need to calculate inverse of X^TXXTX |
| Works well when n is large | Slow if n is very large                         |

With the normal equation, computing the inversion has complexity <i>**O(n<sup>3</sup>)**</i>. So if we have a very large number of features, the normal equation will be slow. **In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.**

## Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of θ even if **<i>(X<sup>T</sup> X)<sup>-1</sup></i>** is noninvertible, the common causes might be having :

- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

## Vectorized Implementations

### Sources

- [Vectorized Implementation in Machine Learning](https://towardsdatascience.com/vectorization-implementation-in-machine-learning-ca652920c55d)
- [Andrew Ng's Vectorized Implementations](https://www.coursera.org/learn/machine-learning/lecture/WnQWH/vectorization)

### Hypothesis Vectorization

Here's our usual hypothesis for linear regression:

![Hypothesis for Linear Regression](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Hypothesis-for%20Linear%20Regression.png)

And if you want to compute h(x), notice that there's a sum on the right. And so one thing you could do is, compute the sum from `j = 0` to `j = n` yourself. Another way to think of this is to think of h(x) as theta transpose X (<i>h<sub>θ</sub>(x) = θ<sup>T</sup>X</i>), and what you can do is, think of this as you are computing this inner product between two vectors where theta is your vector, say, theta 0, theta 1, theta 2. If you have two features, if n equals two, and if you think x as this vector, x0, x1, x2, and these two views can give you two different implementations.

In contrast, here's how you would write a vectorized implementation, which is that you would think of a x and theta as vectors. You just said prediction = theta' * x. You're just computing like so. So instead of writing all these lines of code with a for loop, you instead just have one line of code.

![Hypothesis Implementation: Vectorization 1.1](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Hypothesis-Implementation%20Vectorization%201.1.png)

In order to achieve the hypothesis for all the samples as a list, we use the following array dot product:

![Hypothesis Implementation: Vectorization 1.2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Hypothesis-Implementation%20Vectorization%201.2.png)

The code achievement is pretty easy and clean:

```python
%%time
# matrix format
hypo = X @ theta
>>> Wall time: 0 ns
```

### Cost Function Vectorization

Based on the vectorization of hypothesis, we can easily vectorize the cost function as:

![Cost function Implementation: Vectorization](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Cost-function%20Implementation%20Vectorization.png)

### Cost Function Derivation Vectorization

The derivation of cost function regards to each θ can be vectorized as:

![Cost function Implementation: Vectorization 1.1](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Cost-function%20Implementation%20Vectorization%201.1.png)

The derivation of cost function to all θ can be vectorized as:

![Cost function Implementation: Vectorization 1.2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/images/Cost-function%20Implementation%20Vectorization%201.2.png)

### "Side" Mentions

Read more about optimizations, math, demonstrations, and implementation from the following sources that are worth noting about:

- [Understanding and Calculating the Cost Function for Linear Regression](https://medium.com/@lachlanmiller_52885/understanding-and-calculating-the-cost-function-for-linear-regression-39b8a3519fcb)
- [Gradient Descent in Python](https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f)
