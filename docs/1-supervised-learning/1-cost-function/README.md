# Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the term.

The idea is to choose values that will make the hypothesis function outputs close to the values of y of the training set.

![Cost Function](https://miro.medium.com/max/315/1*E-iWjE3o9luiVapwzYkR7w.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Cost Function Intuition 1](#cost-function-intuition-1)
- [Cost Function Intuition 2](#cost-function-intuition-2)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Cost Function Intuition 1

In simple terms, the cost function controls the variations of the hypothesis function. Our objective is to choose variations that minimizes the value of the cost function (the squared error function or mean squared error), i.e. the cost function is determined by mapping out the average values of the squared differences between the training set and the hypothesis functions to changes in the cost function variables (e.g. the slope). If the idea is to minimize the value of the cost function, then the ideal variation would be mapped to the minimum value of the cost function.

![Cost Function Intuition 1.1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1616284800000&hmac=tq-ZhflWuhdPFKEigeFNpJ9YGLp4W_6JkRMiIx3AjXk)

![Cost Function Intuition 1.2](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1616284800000&hmac=-rZjZ29OUHfNjcYLB8J-0FW3GpgA_vHwd_ub3nGrrfU)

![Cost Function Intuition 1.3](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1616284800000&hmac=WFgw99edPsMpRjW-LETm83krR_kHzueHPw_odaKWAn0)

## Cost Function Intuition 2

A contour plot is a graph that contains many contour lines.

![Contour Plot](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1616284800000&hmac=aZFvZh3HgpHyvMEguEGTR8IObjiTzZmwwaBhY4oKdq4)

A contour line of a two variable function has a constant value at all points of the same line. In certain variations, the value of the cost function in the contour plot gets closer to the center thus reducing the cost function error, giving our hypothesis function a better fit of the data.

![Cost Function Intuition 2.1](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1616284800000&hmac=9sF2SD8FsfYCaqd1tPqHT9qjdutI595UXScnCwmHajM)

![Cost Function Intuition 2.2](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1616284800000&hmac=K7hov6yDNEKeQBBIvn-Dh9MAxKLWhGQf5lB8kIK4JEk)
