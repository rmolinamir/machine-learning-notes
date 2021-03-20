# machine-learning-introduction

> Just random notes while studying ML.

---

## Material

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

## References

> TODO

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

## Model Representation

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h<sub>(x)</sub> is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

## Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the term.

The idea is to choose values that will make the hypothesis function outputs close to the values of y of the training set.

![Cost Function Explanation](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/R2YF5Lj3EeajLxLfjQiSjg_110c901f58043f995a35b31431935290_Screen-Shot-2016-12-02-at-5.23.31-PM.png?expiry=1616284800000&hmac=RBH3USJDGMmB4zb7t0v8gSnbn9-IUezaiyMDPo8kfvg)

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

![Gradient Descent Intuition 1.3](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1616371200000&hmac=niz0t9jgthSbLQoaoRc5L2UhBhBuui0qV09ltM1wUy0)

On a side note, we should adjust our parameter α to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

### Gradient Descent For Linear Regression 

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![Gradient Descent For Linear Regression  Example](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1616371200000&hmac=gVdF9fK14efcORDuwszL6RjzM7GJaH9whFe0wx7VcnU)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.

## Linear Regression with Multiple Variables (or Features)

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

> - x<sub>j</sub><sup>(i)</sup> = value of feature j in the ith training example
> - x<sup>(i)</sup> = the input (features) of the ith training example
> - m = the number of training examples
> - n = the number of features

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

> h<sub>θ</sub>(x) = θ<sup>T</sup>x
