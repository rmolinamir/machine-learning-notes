# Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the term.

The idea is to choose values that will make the hypothesis function outputs close to the values of y of the training set.

![Cost Function](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Cost Function Intuition 1](#cost-function-intuition-1)
- [Cost Function Intuition 2](#cost-function-intuition-2)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Cost Function Intuition 1

In simple terms, the cost function controls the variations of the hypothesis function. Our objective is to choose variations that minimizes the value of the cost function (the squared error function or mean squared error), i.e. the cost function is determined by mapping out the average values of the squared differences between the training set and the hypothesis functions to changes in the cost function variables (e.g. the slope). If the idea is to minimize the value of the cost function, then the ideal variation would be mapped to the minimum value of the cost function.

![Cost Function Intuition 1.1](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function%20Intuition%201.1.png)

![Cost Function Intuition 1.2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function%20Intuition%201.2.png)

![Cost Function Intuition 1.3](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function%20Intuition%201.3.png)

## Cost Function Intuition 2

A contour plot is a graph that contains many contour lines.

![Contour Plot](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Contour-Plot.png)

A contour line of a two variable function has a constant value at all points of the same line. In certain variations, the value of the cost function in the contour plot gets closer to the center thus reducing the cost function error, giving our hypothesis function a better fit of the data.

![Cost Function Intuition 2.1](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function%20Intuition%202.1.png)

![Cost Function Intuition 2.2](https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/1-supervised-learning/1-cost-function/images/Cost-Function%20Intuition%202.2.png)
