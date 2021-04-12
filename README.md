# machine-learning-introduction

> Just random notes while studying ML.

---

## Quick Access Info

- [Stanford Machine Learning Summary](https://www.holehouse.org/mlclass/)

### Images

- [Imgur Album](https://imgur.com/a/mFD48rr)

### Cheatsheets

![Linear Regression Cheat Sheet](https://miro.medium.com/max/1000/1*PZ3TTZZIT1wlqyt05TpZBg.png)
![Logistic Regression Cheat Sheet](https://miro.medium.com/max/1000/1*YNmikbD5k_reqBF1QytErQ.png)

### Theory

- [Professor Andrew Ng's Machine Learning](https://www.coursera.org/learn/machine-learning)
- [Professor Andrew Ng's ML Coursera Python Assignments](https://github.com/dibgerge/ml-coursera-python-assignments)
    > This repositry contains the python versions of the programming assignments for the Machine Learning online class taught by Professor Andrew Ng. This is perhaps the most popular introductory online machine learning class. In addition to being popular, it is also one of the best Machine learning classes any interested student can take to get started with machine learning. An unfortunate aspect of this class is that the programming assignments are in MATLAB or OCTAVE, probably because this class was made before python became the go-to language in machine learning.
- [https://www.deeplearning.ai/](https://www.deeplearning.ai/)

### Practical Applications

- [fast.ai](https://www.fast.ai/)

### Support Sites

- [Reddit's r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning)

---

## Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "Regression" and "Classification" problems.

In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.

In a classification problem, we are instead trying to predict results in a discrete output.

### Example 1

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "Sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

### Example 2

- Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
- Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

## Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

Example: Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. The algorithm tries to find patterns within the chaos.

### Unsupervised Learning Model Representation

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h<sub>(x)</sub> is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

## Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the term.

The idea is to choose values that will make the hypothesis function outputs close to the values of y of the training set.

![Cost Function](https://miro.medium.com/max/315/1*E-iWjE3o9luiVapwzYkR7w.png)

### Cost Function Intuition 1

In simple terms, the cost function controls the variations of the hypothesis function. Our objective is to choose variations that minimizes the value of the cost function (the squared error function or mean squared error), i.e. the cost function is determined by mapping out the average values of the squared differences between the training set and the hypothesis functions to changes in the cost function variables (e.g. the slope). If the idea is to minimize the value of the cost function, then the ideal variation would be mapped to the minimum value of the cost function.

![Cost Function Intuition 1.1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1616284800000&hmac=tq-ZhflWuhdPFKEigeFNpJ9YGLp4W_6JkRMiIx3AjXk)

![Cost Function Intuition 1.2](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1616284800000&hmac=-rZjZ29OUHfNjcYLB8J-0FW3GpgA_vHwd_ub3nGrrfU)

![Cost Function Intuition 1.3](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1616284800000&hmac=WFgw99edPsMpRjW-LETm83krR_kHzueHPw_odaKWAn0)

### Cost Function Intuition 2

A contour plot is a graph that contains many contour lines.

![Contour Plot](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1616284800000&hmac=aZFvZh3HgpHyvMEguEGTR8IObjiTzZmwwaBhY4oKdq4)

A contour line of a two variable function has a constant value at all points of the same line. In certain variations, the value of the cost function in the contour plot gets closer to the center thus reducing the cost function error, giving our hypothesis function a better fit of the data.

![Cost Function Intuition 2.1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1616284800000&hmac=9sF2SD8FsfYCaqd1tPqHT9qjdutI595UXScnCwmHajM)

![Cost Function Intuition 2.2](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1616284800000&hmac=K7hov6yDNEKeQBBIvn-Dh9MAxKLWhGQf5lB8kIK4JEk)

## Gradient Descent

> Here's the idea for gradient descent. What we're going to do is we're going to start off with some initial guesses for &theta;<sub>0</sub> and &theta;<sub>1</sub>. Doesn't really matter what they are, but a common choice would be we set &theta;<sub>0</sub> to 0, and set &theta;<sub>1</sub> to 0. What we're going to do in gradient descent is we'll keep changing &theta;<sub>0</sub> and &theta;<sub>1</sub> a little bit to try to reduce J(&theta;<sub>0</sub>, &theta;<sub>1</sub>), until hopefully, we wind at a minimum, or maybe at a local minimum.

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields &theta;<sub>0</sub> and &theta;<sub>1</sub> (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

![Cost Function of Linear Regression](https://miro.medium.com/max/358/1*vYspf1L6Omqh91nEdHsgEw.png)

We put &theta;<sub>0</sub> on the x axis and &theta;<sub>1</sub> on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific &theta;<sub> </sub>parameters. We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.

![Gradient Descent Graph](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1616284800000&hmac=SBlzqv1K4ZO9Vzp9gPRA4rufVRUUfOmDbiT2cfLe8XY)

### Gradient Descent Algorithm

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of J(&theta;<sub>0</sub>,&theta;<sub>1</sub>). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

Repeat until convergence:

> &theta;<sub>j</sub> := &theta;<sub>j</sub> − α (∂ / ∂&theta;<sub>j</sub>) J(&theta;<sub>0</sub>,&theta;<sub>1</sub>)

Where:

`j = 0, 1`; represents the feature index number.

At each iteration j, one should simultaneously update the parameters &theta;<sub>1</sub>, &theta;<sub>2</sub>, ..., &theta;<sub>n</sub>. Updating a specific parameter prior to calculating another one on the j<sup>(th)</sup> iteration would yield to a wrong implementation.

![Gradient Descent Algorithm Iteration](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1616284800000&hmac=qNIM0mXpzZtOxCYMMxO8fcfei1Otnwl26m-tf_z4xow)

### Gradient Descent Intuition

If the learning rate α is too small, gradient descent can be slow. If the learning rate α is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

Gradient descent can converge to a local minimum, event with the learning rate α fixed. As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease α over time.

![Gradient Descent Intuition 1.1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06.png?expiry=1616371200000&hmac=YuYvPLgF2yIpnD4yWWwpqfbWvGWI_C_3uwByZo0dc5c)

![Gradient Descent Intuition 1.2](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/UJpiD6GWEeai9RKvXdDYag_3c3ad6625a2a4ec8456f421a2f4daf2e_Screenshot-2016-11-03-00.05.27.png?expiry=1616371200000&hmac=wOEjQ-lnOWvbaO7FioPpoVXQMJLYn712IEHj4DCA4jg)

![Gradient Descent Intuition 1.3](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1616544000000&hmac=-fOuuby0bx9OAzNlJ06s8FkJ5VnfjDcXLPo_a0-J36Q)

On a side note, we should adjust our parameter α to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

### Gradient Descent For Linear Regression

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![Gradient Descent For Linear Regression  Example](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1616544000000&hmac=-fOuuby0bx9OAzNlJ06s8FkJ5VnfjDcXLPo_a0-J36Q)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.

## Linear Regression with Multiple Variables (or Features)

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

> - x<sub>j</sub><sup>(i)</sup> = value of feature j in the ith training example
> - x<sup>(i)</sup> = the input (features) of the ith training example
> - m = the number of training examples
> - n = the number of features

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

> h<sub>θ</sub>(x) = θ<sup>T</sup>X

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![Linear Regression with Multiple Variables Equation Comparison](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/MYm8uqafEeaZoQ7hPZtKqg_c974c2e2953662e9578b38c7b04591ed_Screenshot-2016-11-09-09.07.04.png?expiry=1616544000000&hmac=-fOuuby0bx9OAzNlJ06s8FkJ5VnfjDcXLPo_a0-J36Q)

### Gradient Descent in Practice I: Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

![Feature Scaling Range 1](https://latex.codecogs.com/gif.latex?-1%20%5Cleq%20x_%7B%28i%29%7D%5Cleq%201)

![Feature Scaling Range 2](https://latex.codecogs.com/gif.latex?-0.5%20%5Cleq%20x_%7B%28i%29%7D%20%5Cleq%200.5)

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are feature scaling and mean normalization. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

![Feature Scaling Equation](https://latex.codecogs.com/gif.latex?x_%7Bi%7D%20%3A%3D%20%5Cfrac%7Bx_%7Bi%7D%20-%20%5Cmu_%7Bi%7D%7D%7Bs_%7Bi%7D%7D)

Where <i>μ<sub>i</sub></i> is the average of all the values for feature (i) and <i>s<sub>i</sub></i> is the range of values (max - min), or <i>s<sub>i</sub></i> is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if Xi represents housing prices with a range of 100 to 2000  and a mean value of 1000, then:

![Feature Scaling Equation Example](https://latex.codecogs.com/gif.latex?x_%7Bi%7D%20%3A%3D%20%5Cfrac%7Bprice%20-%201000%7D%7B1900%7D)

### Gradient Descent in Practice I: Learning Rate

**Debugging gradient descent:** Make a plot with number of iterations on the x-axis. Now plot the cost function J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

**Automatic convergence test:** Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10<sup>-3</sup>. However in practice it's difficult to choose this threshold value. It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.

To summarize:

- If α is too small: slow convergence.
- If α is too large: may not decrease on every iteration and thus may not converge.

### Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine x<sub>1</sub> and x<sub>2</sub> into a new feature x<sub>3</sub> by taking x<sub>1</sub>⋅ x<sub>2</sub>.

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> then we can create additional features based on x<sub>1</sub>, to get the quadratic function h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>1</sub><sup>2</sup>  or the cubic function h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>1</sub><sup>2</sup> + θ<sub>3</sub>x<sub>1</sub><sup>3</sup>.

In the cubic version, we have created new features x<sub>2</sub> and x<sub>3</sub>, where x<sub>2</sub> = x<sub>1</sub><sup>2</sup> and x<sub>3</sub> = x<sub>1</sub><sup>3</sup>.

To make it a square root function, we could do:

![Features and Polynomial Regression Square Root Function](https://latex.codecogs.com/gif.latex?h_%7B%5CTheta%20%7D%5Cleft%20%28%20x%20%5Cright%20%29%20%3D%20%5CTheta_%7B0%7D%20&plus;%20%5CTheta_%7B1%7Dx_%7B1%7D%20&plus;%20%5CTheta_%7B2%7D%5Csqrt%7Bx_%7B1%7D%7D)

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important. For example, if x<sub>1</sub> has range 1 - 1000 then range of x<sub>1</sub><sup>2</sup> becomes 1 - 1000000 and that of x<sub>1</sub><sup>3</sup> becomes 1 - 1000000000.

### Normal Equation

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

### Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of θ even if **<i>(X<sup>T</sup> X)<sup>-1</sup></i>** is noninvertible, the common causes might be having :

- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

### Vectorized Implementations

#### Sources

- [Vectorized Implementation in Machine Learning](https://towardsdatascience.com/vectorization-implementation-in-machine-learning-ca652920c55d)
- [Andrew Ng's Vectorized Implementations](https://www.coursera.org/learn/machine-learning/lecture/WnQWH/vectorization)

#### Hypothesis Vectorization

Here's our usual hypothesis for linear regression:

![Hypothesis for Linear Regression](https://miro.medium.com/max/377/1*k-sttEqbKIOSlAas3C36CA.png)

And if you want to compute h(x), notice that there's a sum on the right. And so one thing you could do is, compute the sum from `j = 0` to `j = n` yourself. Another way to think of this is to think of h(x) as theta transpose X (<i>h<sub>θ</sub>(x) = θ<sup>T</sup>X</i>), and what you can do is, think of this as you are computing this inner product between two vectors where theta is your vector, say, theta 0, theta 1, theta 2. If you have two features, if n equals two, and if you think x as this vector, x0, x1, x2, and these two views can give you two different implementations.

In contrast, here's how you would write a vectorized implementation, which is that you would think of a x and theta as vectors. You just said prediction = theta' * x. You're just computing like so. So instead of writing all these lines of code with a for loop, you instead just have one line of code.

![Hypothesis Implementation: Vectorization 1.1](https://miro.medium.com/max/539/1*K6bwmo5ZA00aQau-_S4ryg.png)

In order to achieve the hypothesis for all the samples as a list, we use the following array dot product:

![Hypothesis Implementation: Vectorization 1.2](https://miro.medium.com/max/512/1*U4W4RljYFmJQVsLi4YVozw.png)

The code achievement is pretty easy and clean:

```python
%%time
# matrix format
hypo = X @ theta
>>> Wall time: 0 ns
```

#### Cost Function Vectorization

Based on the vectorization of hypothesis, we can easily vectorize the cost function as:

![Cost function Implementation: Vectorization](https://miro.medium.com/max/336/1*rZjirMxeNnyJvziPsOv6Xw.png)

#### Cost Function Derivation Vectorization

The derivation of cost function regards to each θ can be vectorized as:

![Cost function Implementation: Vectorization 1.1](https://miro.medium.com/max/336/1*rZjirMxeNnyJvziPsOv6Xw.png)

The derivation of cost function to all θ can be vectorized as:

![Cost function Implementation: Vectorization 1.2](https://miro.medium.com/max/283/1*TYSV4TecQ9DgSZ_sAQD1mw.png)

#### "Side" Mentions

Read more about optimizations, math, demonstrations, and implementation from the following sources that are worth noting about:

- [Understanding and Calculating the Cost Function for Linear Regression](https://medium.com/@lachlanmiller_52885/understanding-and-calculating-the-cost-function-for-linear-regression-39b8a3519fcb)
- [Gradient Descent in Python](https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f)

---

## Classification

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x<sup>(i)</sup> may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y ∈ {0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given x<sup>(i)</sup>, the corresponding y<sup>(i)</sup> is also called the label for the training example.

### Hypothesis Representation

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense <i>h<sub>θ</sub>(x)</i> to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses <i>h<sub>θ</sub>(x)</i> to satisfy:

> 0 <= <i>h<sub>θ</sub>(x)</i> <= 1

This is accomplished by plugging <i>θ<sup>T</sup>x</i> into the Logistic Function. Our new form uses the "Sigmoid Function," also called the "Logistic Function":

![Sigmoid Function](https://qph.fs.quoracdn.net/main-qimg-6b67bea3311c3429bfb34b6b1737fe0c)

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

<i>h<sub>θ</sub>(x)</i>will give us the probability that our output is 1. For example, <i>h<sub>θ</sub>(x)</i> = 0.7</i> gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

### Logistic Regression Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function. Instead, the cost function will vary depending if the value of y is equal to 1 or 0.

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

Notice that the algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

![Cost Function Vectorized Implementation](https://latex.codecogs.com/gif.latex?%5CTheta%20%3A%3D%5CTheta%20-%5Cfrac%7B%5Calpha%20%7D%7Bm%7DX%5E%7BT%7D%28g%28X%5CTheta%20%29-%5Cvec%7By%7D%29)

### Advanced Optimization Algorithms for Logistic Regresion

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. Do not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized.

### Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}. Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

### Overfitting

Overfitting refers to hypothesis functions that are "overly" fitted to the traning set. These hypothesis functions may have a cost function value very close to zero or zero, but they are generally not accurate when predicting values outside the training zet. In contrast, underfitting hypothesis functions do not capture the data very well.

Underfitting, or high bias, is when the form of our hypothesis function maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:
      - Manually select which features to keep.
      - Use a model selection algorithm (studied later in the course).
2) Regularization:
      - Keep all the features, but reduce the magnitude of parameters <i>θ<sub>j</sub></i>.
      - Regularization works well when we have a lot of slightly useful features.

### Regularization Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

The λ, or lambda, is the regularization parameter and it regulates the theta parameters. It determines how much the costs of our theta parameters are inflated.

Using the cost function with the extra summation of the regularization parameter keeping the theta parameters in check, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if lambda is 0 or is too small? Nothing, but no regularization would be in place.

### Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression.

#### Regularized Linear Regression Gradient Descent

We will modify our gradient descent function to separate out θ<sub>θ</sub> from the rest of the parameters because we do not want to penalize θ<sub>0</sub>.

![Regularized Linear Regression Gradient Descent Equation](https://i.imgur.com/5PKjv22.png)

The term accompanying θ<sub>j</sub> performs our regularization. With some manipulation our update rule can also be represented as:

![Regularized Linear Regression Gradient Descent Equation 2](https://imgur.com/5EAqZO5.png)

The first term in the above equation accompanying lambda, will always be less than 1. Intuitively you can see it as reducing the value of theta by some amount on every update. Notice that the second term is now exactly the same as it was before.

#### Regularized Linear Regression Normal Equation

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

![Regularized Linear Regression Normal Equation](https://imgur.com/TAnyAn2.png)

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including <i>x<sub>0</sub></i>), multiplied with a single real number λ.

### Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![Regularized logistic Regression Normal Equation](https://imgur.com/wAtynOX.png)

![Regularized logistic Regression Normal Equation 2](https://imgur.com/JBSKRKw.png)

---

## Neural Networks

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features, and the output is the result of our hypothesis function.

### Neural Networks Model Representation

In our model, our dendrites are like the input features <i>x<sub>0</sub> ... x<sub>n</sub></i>, and the output is the result of our hypothesis function. In this model our <i>x<sub>0</sub></i> input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our "theta" parameters are sometimes called "weights".

![Neural Networks Model Representation Simplistic Representation](https://i.imgur.com/YvrP11j.png)

- Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".
- We can have intermediate layers of nodes between the input and output layers called the "hidden layers."
- In this example, we label these intermediate or "hidden" layer nodes <i>a<sup>2</sup><sub>0</sub> ... a<sup>2</sup><sub>n</sub></i> and call them "activation units", where the number 2 indicates the layer number, or `j`.

Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, and the dimensions of these matrices of weights is determined as follows:

![Matrix of Weights Explanation](https://i.imgur.com/Np83NLF.png)

The +1 comes from the addition of the "bias nodes". In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:

![Neural Networks Model Representation 1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0rgjYLDeEeajLxLfjQiSjg_0c07c56839f8d6e8d7b0d09acedc88fd_Screenshot-2016-11-22-10.08.51.png?expiry=1617580800000&hmac=OJE8Zcbk40TyD8YYLApFy9Nzs-Ciz9ukHKSSspIBMmM)

### Neural Networks Cost Function

Let's first define a few variables that we will need to use:

- L = total number of layers in the network
- <i>S<sub>l</sub></i> = number of units (not counting bias unit) in layer l
- K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote <i>h<sub>Θ</sub>(x)<sub>k</sub></i> as being a hypothesis that results in the <i>k<sup>th</sup></i> output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

![Neural Networks Cost Function](https://i.imgur.com/9G6EwQJ.png)

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:

- the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
- the triple sum simply adds up the squares of all the individual Θs in the entire network.
- the i in the triple sum does not refer to training example i

### Backpropagation Algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

min<sub>Θ</sub><i>J(Θ)</i>

That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

![Backpropagation partial derivative of J(Θ)](https://i.imgur.com/3FkKjSl.png)

To do so, we use the following algorithm:

Given training set {(<i>x<sup>(1)</sup>, y<sup>(1)</sup></i>) ... (<i>x<sup>(m)</sup>, y<sup>(m)</sup></i>)}

- Set Δ<i><sup>(l)</sup><sub>i,j</sub></i> := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

For training example t =1 to m:

1. Set <i>a<sup>(1)</sup> := x<sup>(t)</sup></i>

2. Perform forward propagation to compute <i>a<sup>(1)</sup></i> for l = 2, 3, …, L.

      ![Forward propagation](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bYLgwteoEeaX9Qr89uJd1A_73f280ff78695f84ae512f19acfa29a3_Screenshot-2017-01-10-18.16.50.png?expiry=1617667200000&hmac=DD9nNkHvYm8XPW2K_xLf2UH3c_cxZioEMEXET6GHSL8)

3. Using <i>y<sup>(t)</sup></i>, compute <i>δ<sup>(L)</sup> = a<sup>(L)</sup> - y<sup>(t)</sup></i>

      Where L is our total number of layers and <i>a<sup>(L)</sup></i> is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute <i>δ<sup>(L - 1)</sup>, δ<sup>(L - 2)</sup>, ..., δ<sup>2</sup></i> using vectorized implementation <i>δ<sup>(l)</sup> = ((Θ<sup>(l)</sup>)<sup>T</sup>δ<sup>(l + 1)</sup>) . *a<sup>(l)</sup> .* (1 - a<sup>(l)</sup>)</i>

   The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by <i>z<sup>(l)</sup></i>.

   The g-prime derivative terms can also be written out as:

   ![g-prime derivative terms](https://i.imgur.com/HwBuykQ.png)

5. We update our new Δ matrix:

   ![Delta Matrix 1](https://i.imgur.com/KM9YoN9.png)
   ![Delta Matrix 2](https://i.imgur.com/zdmRUE3.png)

   The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get:

   ![Delta Matrix 3](https://i.imgur.com/6DlQkns.png)

### Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

![Gradienct Checking 1](https://i.imgur.com/8M2iNR2.png)

With multiple theta matrices, we can approximate the derivative with respect to Θ as follows:

![Gradient Checking 2](https://i.imgur.com/rvTqcxM.png)

A small value for ϵ (epsilon) such as <i>ϵ = 10<sup>-4</sup></i>, guarantees that the math works out properly. If the value for \epsilonϵ is too small, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the Θ matrix. In octave we can do it as follows:

```octave
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;

```

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.

Once you have verified **once** that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow.

### Random Initialization (for Theta parameters)

> When you're running an algorithm of gradient descent, or also the advanced optimization algorithms, we need to pick some initial value for the parameters theta. So for the advanced optimization algorithm, it assumes you will pass it some initial value for the parameters theta.

Initializing all theta weights to zero does not work with neural networks (because all of the hidden units would have the same activation values). When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our Θ matrices using the following method:

![Symmety Breaking](https://i.imgur.com/Gq2BgYn.png)

Hence, we initialize each Θ to a random value between [−ϵ,ϵ]. Using the above formula guarantees that we get the desired bound. The same procedure applies to all the Θ values. Below is some working code you could use to experiment:

```octave
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

rand(x,y) is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.

(Note: the epsilon used above is unrelated to the epsilon from Gradient Checking)

### Putting it Together

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

- Number of input units = dimension of features <i>x<sup>(i)</sup></i>
- Number of output units = number of classes
- Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
- Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

**Training a Neural Network**:

1. Randomly initialize the weights
2. Implement forward propagation to get <i>h<sub>Θ</sub>(x<sup>(i)</sup>)</i> for any <i>x<sup>(i)</sup></i>
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

```octavte
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

The following image gives us an intuition of what is happening as we are implementing our neural network:

![Neural Network Gradient Descent Intuition](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hGk18LsaEea7TQ6MHcgMPA_8de173808f362583eb39cdd0c89ef43e_Screen-Shot-2016-12-05-at-10.40.35-AM.png?expiry=1617840000000&hmac=g-Cgo_Qr2lzT451qlBsRQT_gV99LSpgkSCkfphiQzAc)

Ideally, you want <i>h<sub>Θ</sub>(x<sup>(i)</sup>) ≈ y<sup>(i)</sup></i>. This will minimize our cost function. However, keep in mind that <i>J(Θ)</i> is not convex and thus we can end up in a local minimum instead.

## Diagnostics

Machine Learning Diagnostics is a test you can run, to get insight into what is or isn't working with an algorithm, and which will often give you insight as to what are promising things to try to improve a learning algorithm's performance.

Suppose that after you take your learnt parameters, if you test your hypothesis on the new set of data, suppose you find that this is making huge errors in the predictions. The question is what should you then try mixing in order to improve the learning algorithm? There are many things that one can think of that could improve the performance of the learning algorithm, such as:

- **Getting more training examples:**  Fixes high variance
- **Trying smaller sets of features:** Fixes high variance
- **Trying additional features:** Fixes high bias
- **Trying polynomial features:** Fixes high bias
- **Decreasing λ:** Fixes high bias
- **Increasing λ:** Fixes high variance

Fortunately, there is a pretty simple technique that can let you very quickly rule out half of the things on this list as being potentially promising things to pursue. And there is a very simple technique, that if you run, can easily rule out many of these options, and potentially save you a lot of time pursuing something that's just is not going to work.

### Diagnosing Neural Networks

A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper. A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.
Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

Model Complexity Effects:

- Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
- Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
- In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

### Evaluating a Hypothesis

Once we have done some trouble shooting for errors in our predictions, we can move on to evaluate our new hypothesis. A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a training set and a test set. Typically, the training set consists of 70% of your data and the test set is the remaining 30%.

The new procedure using these two sets is then:

1. Learn Θ and minimize <i>J<sub>train</sub>(Θ)</i> using the training set
2. Compute the test set error <i>J<sub>test</sub>(Θ)</i>

To computer the test set error:

![The test set error](https://i.imgur.com/qBLc8s1.png)

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

![Test Error](blob:https://imgur.com/257d6e08-f397-43a6-9f21-925b6a09eef7)

This gives us the proportion of the test data that was misclassified.

### Model Selection and Train/Validation/Test Sets

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

### Diagnosing Bias vs. Variance

In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.

We need to distinguish whether bias or variance is the problem contributing to bad predictions.
High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.
The training error will tend to decrease as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to decrease as we increase d up to a point, and then it will increase as d is increased, forming a convex curve.

High bias (underfitting): both the training and the cross validation errors will be high. Also, the training error will be approximately equal to the cross validation error.

High variance (overfitting): the training error is low, but the cross validation error will be much greater than the training error.

The is summarized in the figure below:

![Diagnosing Bias vs. Variance](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/I4dRkz_pEeeHpAqQsW8qwg_bed7efdd48c13e8f75624c817fb39684_fixed.png?expiry=1617926400000&hmac=VZoumHH6rUt2FxeDXqVxBXWUH3FjZITom9G117kPRZ0)

### Regularization and Bias/Variance

How do we choose our parameter λ to get it 'just right'? In order to choose the model and the regularization term λ, we need to:

1. Create a list of lambdas (i.e. λ ∈ {0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
2. Create a set of models with different degrees or any other variants.
3. Iterate through the λs and for each λ go through all the models to learn some Θ.
4. Compute the cross validation error using the learned Θ (computed with λ) on the cross validation cost function **without** regularization or λ = 0.
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo Θ and λ, apply it on the test cost function and calculate the error to see if it has a good generalization of the problem.

![Regularization and Bias/Variance](https://i.imgur.com/TKIIMuz.png)

### Learning Curves

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

### Neural Networks and Overfitting

- Small neural networks (fewer parameters) are more prone to underfitting, but it is computationally cheaper.
- Large neural networks (more parameters) are more prone to overfitting, but it is computationally more expensive. Use regularization to address overfitting.

## Prioritizing What to Work On

Given a data set of emails, we could construct a vector for each email. Each entry in this vector represents a word. The vector normally contains 10,000 to 50,000 entries gathered by finding the most frequently used words in our data set.  If a word is to be found in the email, we would assign its respective entry a 1, else if it is not found, that entry would be a 0. Once we have all our x vectors ready, we train our algorithm and finally, we could use it to classify if an email is a spam or not.

So how could you spend your time to improve the accuracy of this classifier?

- Collect lots of data (for example "honeypot" project but doesn't always work)
- Develop sophisticated features (for example: using email header data in spam emails)
- Develop algorithms to process your input in different ways (recognizing misspellings in spam).

It is difficult to tell which of the options will be most helpful.

### Error Analysis

The recommended approach to solving machine learning problems is to:

- Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
- Plot learning curves to decide if more data, more features, etc. are likely to help.
- Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.

For example, assume that we have 500 emails and our algorithm misclassifies a 100 of them. We could manually analyze the 100 emails and categorize them based on what type of emails they are. We could then try to come up with new cues and features that would help us classify these 100 emails correctly. Hence, if most of our misclassified emails are those which try to steal passwords, then we could find some features that are particular to those emails and add them to our model. We could also see how classifying each word according to its root changes our error rate.

It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance. For example if we use stemming, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), and get a 3% error rate instead of 5%, then we should definitely add it to our model. However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature.  Hence, we should try new things, get a numerical value for our error rate, and based on our result decide whether we want to keep the new feature or not.

## Handling Skewed Data

In the context of evaluation and of error metrics, there is one important case, where it's particularly tricky to come up with an appropriate error metric, or evaluation metric, for your learning algorithm, that case is the case of what's called skewed classes. Consider the problem of cancer classification, where we have features of medical patients and we want to decide whether or not they have cancer. Let's say y equals 1 if the patient has cancer and y equals 0 if they do not. We have trained the progression classifier and let's say we test our classifier on a test set and find that we get 1 percent error.

So only half a percent of the patients that come through our screening process have cancer. In this case, the 1% error no longer looks so impressive. When we're faced with such a skewed classes therefore we would want to come up with a different error metric than accuracy or a different evaluation metric. One such evaluation metric are what's called precision recall.

### Error Metrics for Skewed Classes

Let's say you have one joining algorithm that's getting 99.2% accuracy, so, that's a 0.8% error. Let's say you make a change to your algorithm and you now are getting 99.5% accuracy, that is 0.5% error. So, is this an improvement to the algorithm or not? Did we just do something useful or did we just replace our code with something that just predicts y equals zero more often? So, if you have very skewed classes it becomes much harder to use just classification accuracy, because you can get very high classification accuracies or very low errors, and it's not always clear if doing so is really improving the quality of your classifier because predicting y equals 0 all the time doesn't seem like a particularly good classifier.

Precision and recall error metrics that give us a better sense of how well our classifier is doing. For example, if we have a learning algorithm that predicts y equals zero all the time then this classifier will have a recall equal to zero, because there won't be any true positives and so that's a quick way for us to recognize that a classifier that predicts y equals 0 all the time, just isn't a very good classifier.

For the problem of skewed classes precision recall gives us more direct insight into how the learning algorithm is doing and this is often a much better way to evaluate our learning algorithms, than looking at classification error or classification accuracy, when the classes are very skewed.

Imagine a two by two table as follows, depending on a full of these entries depending on what was the actual class and what was the predicted class:

![Error Metrics for Skewed Classes](https://i.imgur.com/jjhWMcV.png)

A true positive means our algorithm predicted that it's positive and in reality the example is positive.
A true negative means our algorithm predicted that something is negative, class zero, and the actual class is also class zero.
A false positive means our algorithm predicts that the class is one but the actual class is zero.
A false negative means our algorithm predicted zero, but the actual class was one.

But what is precision and recall?

Precision is the ratio between true positive among all predicted positives. For example, for all the patients that were told, "We think you have cancer", of all those patients, what fraction of them actually have cancer is the precision.

Recall is the ratio between true positves among all actual positives. For example, out of all patients, what is the right number of actual positives of all the people that do have cancer. What fraction do we directly flag as having cancer, then advice for treatment.

### Trading Off Precision and Recall

For many applications, we'll want to somehow control the trade-off between precision and recall.

The F Score, which is also called the F1 Score, is a little bit like taking the average of precision and recall, but it gives a higher weight to the lower value of precision and recall, whichever it is. And so, you see in the numerator here that the F Score takes a product of precision and recall. And so if either precision is 0 or recall is equal to 0, the F Score will be equal to 0. So in that sense, it kind of combines precision and recall, but for the F Score to be large, both precision and recall have to be pretty large. I should say that there are many different possible formulas for combing precision and recall. This F Score formula is really just one out of a much larger number of possibilities, but historically or traditionally this is what people in Machine Learning seem to use:

![Trading Off Precision and Recall F1 Score](blob:https://imgur.com/c9bb13c9-da81-4563-ae58-25305514fe7a)

And the term F Score, it doesn't really mean anything, so don't worry about why it's called F Score or F1 Score. The F Score is used to use precision and recall as an evaluation metric for learning algorithms, more specifically, as a single real number evaluation metric. So you try a range of values of thresholds and evaluate these different thresholds on your cross-validation set and then to pick whatever value of threshold gives you the highest F Score on your cross validation data set. That would be a pretty reasonable way to automatically choose the threshold for your classifier as well.

### Data For Machine Learning

> So, if you have a lot of data and you train a learning algorithm with lot of parameters, that might be a good way to give a high performance learning algorithm, the key test that I often ask myself are first, can a human experts look at the features x and confidently predict the value of y? Because that's sort of a certification that y can be predicted accurately from the features x and second, can we actually get a large training set, and train the learning algorithm with a lot of parameters in the training set and if you can't do both then that's more often give you a very kind performance learning algorithm.

Under certain conditions, getting a lot of data and training on a certain type of learning algorithm, can be a very effective way to get a learning algorithm to do very good performance. This arises often enough that if those conditions hold true for your problem and if you're able to get a lot of data, this could be a very good way to get a very high performance learning algorithm.

Consider a problem of predicting the price of a house from only the size of the house and from no other features. Imagine I tell you that a house is, 500 square feet but I don't give you any other features. I don't tell you that the house is in an expensive part of the city. Or if I don't tell you that the house, the number of rooms in the house, or how nicely furnished the house is, or whether the house is new or old. If I don't tell you anything other than that the house is a 500 square foot house, there's so many other factors that would affect the price of a house other than just the size of a house that if all you know is the size, it's actually very difficult to predict the price accurately.

If we were to go to human expert in this domain, can experts actually confidently predict the value of y? For this first example if we go to an expert realtor and just tell them the size of a house and I tell them what the price is, even an expert in pricing of houses wouldn't be able to tell me knowing only the size doesn't provide enough information to predict the price of the house.

But, when having a lot of data could help? Suppose the features have enough information to predict the value of y. And let's suppose we use a learning algorithm with a large number of parameters, chances are, if we run these algorithms on the data sets, it will be able to fit the training set well, and so hopefully the training error will be slow. Let's say, we use a massive, massive training set, then hopefully even though we have a lot of parameters the training set is much larger than the number of parameters, in thes case the algorithm will be unlikely to overfit which means is that the training error will hopefully be close to the test error, which means the test error will be small.

## Support Vector Machines

### Optimization Objective

There's one more algorithm that is very powerful and is very widely used both within industry and academia, and that's called the SVM (support vector machine).

SVM sometimes gives a cleaner, and sometimes more powerful way of learning complex non-linear functions. The SVM algorithm is essentially a modified logistic regression algorithm. The SVM provides different way of prioritizing how much we care about optimizing the first term in logistic regression, versus how much we care about optimizing the second term. Unlike logistic regression, the support vector machine doesn't output the probability, it outputs a prediction of y being equal to one or zero, directly.

### Large Margin Intuition

Sometimes people refer to SVM as large margin classifiers. We'll consider what that means and what an SVM hypothesis looks like.

The SVM cost function is as below:

![SVM ](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[11].png)

Unlike logistic, <i>h<sub>θ</sub>(x)</i> doesn't give us a probability, but instead we get a direct prediction of 1 or 0
So if <i>θ<sup>T</sup>X</i> is equal to or greater than 0, then <i>h<sub>θ</sub>(x) = 1</i>, otherwise <i>h<sub>θ</sub>(x) = 0</i>. For logistic regression we had two terms;

- Training data set term (i.e. that we sum over m) = A
- Regularization term (i.e. that we sum over n) = B

So we could describe it as A + λB, and it needs some way to deal with the trade-off between regularization and data set terms by setting different values for λ to parametrize this trade-off. Instead of parameterization this as A + λB, for SVMs the convention is to use a different parameter called C, resulting in CA + B, If C were equal to 1/λ then the two functions (CA + B and A + λB) would give the same value.

The SVM cost function is as above, and we've drawn out the cost terms below:

![SVM Cost Terms](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[12].png)

Left is cost<sub>1</sub> and right is cost<sub>0</sub>. If you have a positive example, you only really need z to be greater or equal to 0, if this is the case then you predict 1. The SVM wants a bit more than that - doesn't want to *just* get it right, but have the value be quite a bit bigger than zero to throw in an extra safety margin factor, logistic regression does something similar.

Consider a case where we set C to be huge number, such as C = 100,000. Considering we're minimizing CA + B, if C is huge we're going to pick an A value so that A is equal to zero, wat is the optimization problem here - how do we make A = 0? Making A = 0, if y = 1, then to make our "A" term 0 we need to find a value of θ so (<i>θ<sup>T</sup>X</i>) is greater than or equal to 1. Similarly, if y = 0, then we want to make "A" = 0 then we need to find a value of θ so (<i>θ<sup>T</sup>X</i>) is equal to or less than -1. If we think of our optimization problem a way to ensure that this first "A" term is equal to 0, we re-factor our optimization problem into just minimizing the "B" (regularization) term, because  when A = 0,  A*C = 0. Turns out when you solve this problem you get interesting decision boundaries:

![SVM Decision Boundaries](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[14].png)

- The green and magenta lines are functional decision boundaries which could be chosen by logistic regression, but they probably don't generalize too well.
- The black line, by contrast is the the chosen by the SVM because of this safety net imposed by the optimization graph. It's a more robust separator. Mathematically, that black line has a larger minimum distance (margin) from any of the training examples.

By separating with the largest margin you incorporate robustness into your decision making process, we looked at this at when C is very large. The SVM is more sophisticated than the large margin might look, but if you were just using large margin then SVM would be very sensitive to outliers because you would risk making a ridiculous hugely impact your classification boundary where a single example might not represent a good reason to change an algorithm. If C is very large then we do use this quite naive maximize the margin approach.

### Mathematics Behind Large Margin Classification

Consider the training example below:

![Mathematics Behind Large Margin Classification Example I](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[32].png)

Given this data, what boundary will the SVM choose? Note that we're still assuming <i>θ<sub>0</sub> = 0</i>, which means the boundary has to pass through the origin (0,0):

![Mathematics Behind Large Margin Classification Example II](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[36].png)

SVM would not chose this line because the decision boundary comes very close to examples. **θ is always at 90 degrees to the decision boundary (can show with linear algebra, although we're not going to)**. After projecting a line from <i>x<sup>(1)</sup></i> on to to the θ vector (so it hits at 90 degrees), the distance between the intersection and the origin is (<i>p<sup>1</sup></i>) (the red line). Similarly, the second example (<i>x<sup>(2)</sup></i>) projects a line from <i>x<sup>2</sup></i> into to the θ vector. This is the magenta line, which will be negative (<i>p<sup>2</sup></i>).

We find that both these p values are going to be pretty small. We know we need <i>p<sup>1</sup> \* ||θ||</i> to be bigger than or equal to 1 for positive examples, for this reason, if p is small then that means that ||θ|| must be pretty large. Similarly, for negative examples we need <i>p<sup>2</sup> \* ||θ||</i> to be smaller than or equal to -1. We saw in this example <i>p<sup>2</sup></i> is a small negative number, so ||θ|| must be a large number. Why is this a problem?

The optimization objective is trying to find a set of parameters where the norm of theta is small. So this doesn't seem like a good direction for the parameter vector (because as p values get smaller ||θ|| must get larger to compensate), so we should make p values larger which allows ||θ|| to become smaller.

So lets chose a different boundary:

![Mathematics Behind Large Margin Classification Example III](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[39].png)

Now if you look at the projection of the examples to θ we find that <i>p<sup>1</sup></i> becomes large and ||θ|| can become small. This means that by choosing this second decision boundary we can make ||θ|| smaller, which is why the SVM choses this hypothesis as better.

### Kernels I

Suppose we have a training set and we want to find a non-linear boundary such as the one in the following image:

![Kernels I Example I](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[40].png)

A way of writing this boundary would be to use a hypothesis that computes a decision boundary by taking the sum of the parameter vector multiplied by a new feature vector f, which simply contains the various high order x terms:

<i>h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub></i>

Where:

- <i>f<sub>1</sub>= x<sub>1</sub></i>
- <i>f<sub>2</sub> = x<sub>1</sub>x<sub>2</sub></i>
- ...

These new features can be defined as similarity functions between x values and landmark values. Have a graph of <i>x<sub>1</sub></i> vs. <i>x<sub>2</sub></i>, then pick three points in that space:

![Kernels I Example II](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[41].png)

These points l<sub>1</sub>, l<sub>2</sub>, and l<sub>3</sub>, which were chosen manually and are called landmarks. Given x, define f<sub>1</sub> as the similarity between (x, l<sub>1</sub>).

The similarity function between x and the landmarks can be defined as:

![Kernels I Example III](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[42].png)

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

![Kernels I Example IV](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[45].png)

- Notice that when x = [3,5] then f1 = 1.
- As x moves away from [3,5] then the feature takes on values close to zero.
- This measures how close x is to this landmark.

σ<sup>2</sup> is a parameter of the Gaussian kernel and it defines the steepness of the rise around the landmark. We see here that as you move away from [3,5] the feature f1 falls to zero much more slowly if σ<sup>2</sup> is equal to 3.

![Kernels I Example V](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[47].png)

If each feature of our hypothesis is a similarity function between the x values and the landmark values, then we can draw a decision boundary, when for example: <i>θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0</i>

For our example, lets say we've already run an algorithm and got the parameters:

- <i>θ<sub>0</sub> = -0.5</i>
- <i>θ<sub>1</sub> = 1</i>
- <i>θ<sub>2</sub> = 1</i>
- <i>θ<sub>3</sub> = 0</i>

Imagine a point close to f1. We know being close to f1 will result in a value close to 1 because of the Gaussian kernel, but the values of f2 and f3 will be close to 0. So if we look at the formula we have, <i>θ<sub>0</sub> + θ<sub>1</sub>f<sub>1</sub> + θ<sub>2</sub>f<sub>2</sub> + θ<sub>3</sub>f<sub>3</sub> >= 0</i>, -0.5 + 1 + 0 + 0 = 0.5, 0.5 being greater than 0, so we predict 1. If we had another point far away from all three, it would equate to -0.5, so we predict 0.

Considering our parameter, for points near l1 and l2 you predict 1, but for points near l3 you predict 0. Which means we create a non-linear decision boundary that goes like this:

![Kernels I Example VI](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[50].png)

Where:

- Inside we predict y = 1.
- Outside we predict y = 0.

So this shows how we can create a non-linear boundary with landmarks and the kernel function in the support vector machine.

### Kernels II

How do we get/chose the landmarks? What other kernels can we use (other than the Gaussian kernel)?

#### Choosing the landmarks for Gaussian kernels

Given some training data with `m` examples, for each example in the training data, place a landmark at exactly the same location, so we end up with `m` landmarks. If we have one landmark per training example, this means our features measure how close an input would be to a training example.

So given a new example, compute all of the f feature values or f vector, where:

- <i>f<sub>o</sub> = 1</sup></i> always
- <i>f<sub>1</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>1</sup>)</i>
- <i>f<sub>2</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>2</sup>)</i>
- ...
- <i>f<sub>m</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>m</sup>)</i>

When using the Gaussian kernel, somewhere along the line, we will get a f value equal to: <i>f<sub>i</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>i</sup>)</i>, where <i>x<sup>i</sup></i> and <i>l<sup>i</sup></i> would be identical, being equal to e<sup>-0</sup> = 1. We take these `m` f features (fo, f1, f2 ... fm) to compute a [m+1,1] `f` vector.

After obtaining the f vector, the lambda parameters can then be computed using the minimum cost function. Similarly to logistic regression when y = 1 means <i>θ<sup>T</sup>X >= 0</i>, when y = 1 means <i>θ<sup>T</sup>f >= 0</i>. That being said, the minimum cost function of the SVM learning algorithm would be:

![SVM Cost Function](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[51].png)

In this setup m = n, because number of features is the number of training data examples, and now the cost can be computed using f as the feature vector instead of x.

One final mathematic detail, if we ignore <i>θ<sub>0</sub></i> then the following is true:

![SVM Cost Function Theta Term](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[52].png)

But what many implementations do, is instead use this vectorization of theta:

![SVM Cost Function Theta Vectorization](https://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[52].png)

Where the matrix M depends on the kernel (e.g. Gaussian kernel). This results in a slightly different minimization or a rescaled version of θ. The reason this is done is because it results in a more efficient computation of the minimum costs for optimum theta parameters, and scale well to much bigger training sets. If you have a training set with 10000 values, then that means you get 10000 features. Solving for all these parameters can become expensive, so by adding this in we avoid a for loop and use a matrix multiplication algorithm instead.

You can apply kernels to other algorithms but they tend to be very computationally expensive such as in linear regression algorithms.

#### Choosing parameters C and σ<sup>2</sup> for Gaussian kernels

The parameters C and σ<sup>2</sup> control the bias and variance of the model, this means that varying C and σ<sup>2</sup> results in a trade-off between bias and variance.

**Parameter C:**: C plays a role similar to 1/λ (where λ is the regularization parameter).

- Large C gives a hypothesis of low bias high variance (overfitting).
- Small C gives a hypothesis of high bias low variance (underfitting).

**Parameter σ<sup>2</sup>**: commonly called the variance and it defines the steepness of the rise around the landmark.

- Large σ<sup>2</sup> makes f features vary more smoothly, i.e. higher bias, lower variance.
- Small σ<sup>2</sup> makes f features vary abruptly, i.e. low bias, high variance.

### Using An SVM

It is recommended to use SVM software packages (e.g. liblinear, libsvm) to solve parameters θ, because these packages come with numerical optimizations to calculate optimal θ. However, the parameter C and the kernel function that will be used (e.g. the Gaussian kernel) must be specified.

#### Choosing a kernel

Some kernels used in SVMs are:

- Gaussian kernel
- Linear kernel (no kernel)
- Polynomial kernel
- Chi-squared kernel
- Histogram intersection kernel

Linear and Gaussian kernels are the most common. Not all similarity functions are valid kernels because they must satisfy Merecer's Theorem. SVM use numerical optimization tricks, this means certain optimizations can be made, but they must follow the theorem.

Some SVM packages will expect you to define kernel, although, some SVM implementations include the Gaussian and a few others. Gaussian is probably the most popular kernel.

##### Gaussian kernel

- Need to define σ (σ<sup>2</sup>)
- When would you chose a Gaussian kernel?
  - If `n` is small and/or `m` is large, e.g. 2D training set that's large.
- If you're using a Gaussian kernel then you may need to implement the kernel function, e.g. a function <i>f<sub>m</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>m</sup>)</i> which returns a real number.
- Make sure you perform **feature scaling** before using a Gaussian kernel. If you don't, features with a larger values will lead to other features with smaller values to be ignored and made irrelevant to the prediction.

##### Linear kernel

- Predicts y = 1 when <i>θ<sup>T</sup>X >= 0</i>, this means that there is no no f vector, so a standard linear classifier is used.
- Why do this?
  - If n is large and m is small then this means that there are lots of features, but few examples. This means that there is not enough data, so there is a risk of overfitting in a high dimensional feature-space.

Logistic regression and SVM with a linear kernel are pretty similar. They do similar computations and feature similar performance.

##### Polynomial kernel

- Used when x and l are both non-negative
- We measure the similarity of x and l by computing <i>(X<sup>T</sup>L + ε)<sup>d</sup></i>
  - Where:
    - ε is a constant value.
    - d is the deegree of polynomial.
- This means that if X and l are similar, then the inner product tends to be large.
- This is not used that often because other kernels such as the Gaussian kernel tend to be more effective.

##### String kernel

- Used if input is text strings.
- Use for text classification.

#### Multi-class classification for SVM

Many packages have built in multi-class classification packages, otherwise the one-vs all method can be used, where a number of K theta parameters are calculated where K is the amount of classes, just like in logistic regression.

#### When to use Logistic Regression or SVM?

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
