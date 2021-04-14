# Neural Networks

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features, and the output is the result of our hypothesis function.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Neural Networks Model Representation](#neural-networks-model-representation)
- [Neural Networks Cost Function](#neural-networks-cost-function)
- [Backpropagation Algorithm](#backpropagation-algorithm)
- [Gradient Checking](#gradient-checking)
- [Random Initialization (for Theta parameters)](#random-initialization-for-theta-parameters)
- [Putting it Together](#putting-it-together)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Neural Networks Model Representation

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

## Neural Networks Cost Function

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

## Backpropagation Algorithm

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

## Gradient Checking

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

## Random Initialization (for Theta parameters)

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

## Putting it Together

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
