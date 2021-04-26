# Large Scale Machine Learning

Machine learning works best when there is an abundance of data to leverage for training. With the amount data that many websites/companies are gathering today, knowing how to handle ‘big data’ is one of the most sought after skills in Silicon Valley.

One of the reasons that learning algorithms work so much better is just the sheer amount of data that we have now and that we can train our algorithms on. We'll talk about algorithms for dealing when we have such massive data sets.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

- [Learning With Large Datasets](#learning-with-large-datasets)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
  - [Mini-Batch Gradient Descent Algorithm](#mini-batch-gradient-descent-algorithm)
- [Stochastic Gradient Descent Convergence](#stochastic-gradient-descent-convergence)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Learning With Large Datasets

Learning with large data sets comes with its own unique problems, specifically, computational problems. Let's say your training set size is M equals 100,000,000. And this is actually pretty realistic for many modern data sets. A data set this large gives birth to new and exciting problems that need to be dealt with on both the algorithmic and architectural level of the system.

In a prior section, we delved into a study that concluded with the idea that so long as you feed multiple algorithms with large amounts of data, they end up performing very similarly. And so it's results like these that has led to the saying in machine learning that *often it's not who has the best algorithm that wins, it's who has the most data*.

![Learning With Large Datasets Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image.png)

But learning with large datasets comes with its own computational problems, fFor example, say we have a data set where m = 100.000.000. This is pretty realistic for many modern data sets such as cnesus data, or website traffic data. **How do we train a logistic regression model on such a big system?**

![Learning With Large Datasets Example II](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[1].png)

In the typical gradient descent, you would have to sum over 100.000.000 examples per iteration of the gradient descent. Because of the computational cost of this massive summation, we'll look at more efficient ways around this by either:

- Using a different approach.
- Optimizing to avoid the global summation.

First thing to do is ask if we can train on 1000 examples instead of 100.000.000, to do this we randomly pick a small selection and verify if we can develop a system which performs just as well.

If this is the case you can avoid a lot of the headaches associated with big data, to see if taking a smaller sample works, you can sanity check by plotting the error vs. training set size.

For example, if our plot looked like this:

![Learning With Large Datasets Example III](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[3].png)

- This looks like a high bias problem.
- More examples may not actually help, it would save a lot of time and effort if we know this before hand.
- One natural thing to do here might be to:
  - Add extra features.
  - Add extra hidden units (if using neural networks).

But, if our plot looked like this:

![Learning With Large Datasets Example IV](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[2].png)

- Looks like a high variance problem, which means that the model is overfitted to the training set but is bad at predicting examples outside the training set.
- More examples should improve performance, but large data sets means we have to deal with a new set of problems.

For many learning algorithms, we derived them by coming up with an optimization objective (cost function) and using an algorithm to minimize that cost function.

When you have a large dataset, gradient descent becomes very expensive, so we'll define a different way to optimize for large data sets which will allow us to scale the algorithms.

Suppose you're training a linear regression model with gradient descent:

- **Hypothesis:**
  ![Learning With Large Datasets V](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[4].png)
- **Cost function:**
  ![Learning With Large Datasets VI](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[5].png)
- If we plot our two parameters vs. the cost function we get something like a bowl shaped surface plot:
![Learning With Large Datasets VII](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[6].png)

Remember that gradient descent works by looping over a number of iterations that repeatedly update the parameters θ.

![Learning With Large Datasets VIII](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[7].png)

But in contrast, let's apply a stochastic gradient descent to our linear regression example (the ideas apply to other algorithms too such as Logistic Regression and Neural Networks).

Below we have a contour plot for a typical gradient descent showing iteration to a global minimum:

![Learning With Large Datasets IX](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[8].png)

As mentioned, if m is large, gradient descent can be very computationally expensive. Although so far we just referred to it as gradient descent, this kind of gradient descent is called **batch gradient descent**, which just means we look at all the examples at the same time.

Batch gradient descent is not great for large data sets. For example:

- If you have 300.000.000 records you need to read in all the records into memory.
- After reading all the records, you can move one iteration step (iteration) through the algorithm.
- Then repeat this for EVERY iteration step.
- This means it take a LONG time to converge. This is a system bottleneck and will inevitably require a huge number of reads.

What we're going to do here is come up with a different algorithm which only needs to look at a single example at a time.

## Stochastic Gradient Descent

To apply stochastic gradient descent, let's define our cost function slightly differently:

![Stochastic Gradient Descent Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[9].png)

The function represents the cost of θ with respect to a specific example (x<sup>i</sup>, y<sup>i</sup>), and we calculate this value as one half times the squared error on that example.

It measures how well the hypothesis works on *a single example*. Thus, the overall cost function can now be re-written in the following form:

![Stochastic Gradient Descent Example II](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[10].png)

This is equivalent to the batch gradient descent cost function. With this slightly modified (but equivalent) view of linear regression we can write out how stochastic gradient descent works:

1. Randomly shuffle the data set.
  ![Stochastic Gradient Descent Example III](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[11].png)
2. For each example, compute the gradient and perform a gradient descent step towards the global optima, then update θ, and finally this process is repeated for a few iterations (but sometimes even 1 might be enough, typically it's done 1-10 times).
  ![Stochastic Gradient Descent Example IV](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[12].png)

Let's break this down.

- The *randomly shuffling* at the start means we ensure the data is in a random order so we don't bias the descent towards global optima.
  - Randomization should speed up convergence a little bit.
- The term:
  ![Stochastic Gradient Descent Example V](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[13].png)
  - Is the same as that found in the summation for batch gradient descent.
  - It's possible to show that this term is equal to the partial derivative with respect to the parameter θ<sub>j</sub> of the cost(θ, (x<sup>i</sup>, y<sup>i</sup>)) function.
    ![Stochastic Gradient Descent Example VI](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[14].png)
- What the stochastic gradient descent algorithm is doing is going through each example one at a time while updating θ, the inner loop is doing the following:
  - Looking at example 1, take a step with respect to the cost of just the 1st training example.
  - Having done this, we go on to the second training example.
    - Now take a second step to try and fit the second training example better.
  - And so on until it gets to the end of the data.
- We may now repeat this who procedure and take multiple passes over the data.

Although stochastic gradient descent is a lot like batch gradient descent, rather than waiting to sum up the gradient terms over all m examples, we take just one example and make progress in improving the θ parameters.

This means we update the parameters on EVERY step through data, instead of at the end of each loop through all the data. What does the algorithm do to the parameters?

As we saw, batch gradient descent does something like this to get to a global minimum:

![Stochastic Gradient Descent Example VII](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[15].png)

With stochastic gradient descent every iteration is much faster, but every iteration is flitting a single example:

![Stochastic Gradient Descent Example VIII](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[16].png)

- What you find is that you "generally" move in the direction of the global minimum, but not always.
- It also never converges like the batch gradient descent does, but ends up wandering around some region close to the global minimum.
  - In practice, this isn't a problem, as long as you're close to the minimum.
  - It's also possible to make it converge by reducing the learning rate over each iteration, but it's rather complicated and requires more tinkering with the parameters.

As mentioned, we may need to loop over the entire dataset 1-10 times. If you have a truly massive dataset it's possible that by the time you've taken a single pass through the dataset you may already have a perfectly good hypothesis, in which case **the inner loop might only need to be done once, if m is very very large**.

If we contrast this to batch gradient descent, we have to make k passes through the entire dataset, where k is the number of steps needed to move through the data.

## Mini-Batch Gradient Descent

Mini-batch gradient descent is an additional approach which can work even faster than stochastic gradient descent.

To summarize our approaches so far:

- Batch gradient descent: Use all m examples in each iteration.
- Stochastic gradient descent: Use 1 example in each iteration.
- Mini-batch gradient descent: Use b examples in each iteration.
  - Where b < m, and is the mini-batch size.

It's just like the batch gradient descent, except we use tiny batches. A typical range for b is 2-100 (10 is a common value).

### Mini-Batch Gradient Descent Algorithm

![Mini-Batch Gradient Descent Algorithm Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[17].png)

- We for-loop through b-size batches of m.
- Compared to batch gradient descent this allows us to get through data in a much more efficient way.
  - After just b examples we begin to improve our parameters.
  - We don't have to update parameters after every example, and we don't have to wait until the algorithm cycles through all the data.

Why should we use mini-batch?

- It allows you to have a vectorized implementation.
- This makes the implementation be much more efficient.
- It can partially parallelize your computation (i.e. do 10 examples at once).

A disadvantage of mini-batch gradient descent is the optimization of the parameter b, but the time investment is often worth it.

The stochastic gradient descent and the mini-batch gradient descent are just specific forms of batch gradient descent.

## Stochastic Gradient Descent Convergence

We now know about stochastic gradient descent, to know when is the algorithm "done" we have to check for convergence, and to do this we have to tune the learning rate alpha (α) parameter.

With batch gradient descent, we could plot the cost function vs. number of iterations knowing it should decrease on every iteration. This works when the training set size was small because we could sum over all examples, but this approach doesn't work when you have a massive dataset.

With stochastic gradient descent, we don't want to have to pause the algorithm periodically to do a summation over all data. Moreover, **the whole point of stochastic gradient descent is to avoid those whole-data summations**.

For stochastic gradient descent, we have to do something different:

- Take cost function definition where we have one half the squared error on a single example:
  ![Stochastic Gradient Descent Convergence Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[18].png)
  - While the algorithm is looking at the example (x<sup>i</sup>, y<sup>i</sup>) we can compute the cost of the example (cost(θ, (x<sup>i</sup>, y<sup>i</sup>))) before it has updated θ.
    - In other words, we compute how well the hypothesis is working on the training example, **but we eed to do this before we update θ**.  If we did it *after θ* was updated the algorithm would be performing a bit better when computing the cost.
  - To check for the convergence, every 1000 iterations we can plot the costs averaged over the last 1000 examples.
    - This gives a running estimate of how well we've done on the last 1000 estimates.
    - By looking at the plots we should be able to check for convergence.

These plots would look like the pictures below. In general, the plots might be a bit noisy (as 1000 examples isn't that much).

If you get a figure like this:

![Stochastic Gradient Descent Convergence Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[19].png)

- This plot indicates a pretty decent run.
- The algorithm may have reached convergence.

If you use a smaller learning rate you may get an even better final solution:

![Stochastic Gradient Descent Convergence Example II](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[20].png)

- This is because the parameter oscillates around the global minimum.
- A smaller learning rate means smaller oscillations.

If you average over 1000 examples and 5000 examples you may get a smoother curve:

![Stochastic Gradient Descent Convergence Example III](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[20].png)

- This disadvantage of a larger average means you get less frequent feedback

![Stochastic Gradient Descent Convergence Example IV](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[21].png)

- It looks like the cost is not decreasing at all (blue line).
- But if you then increase to average over a larger number of examples you do see this general trend (red line).
  - The blue line was too noisy, and that noise is ironed out by taking a greater number of entires per average.
- It may not decrease, even with a large number.

If you see a curve the looks like its increasing then the algorithm may be displaying divergence:

![Stochastic Gradient Descent Convergence Example V](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[23].png)

- Should use a smaller learning rate.

We saw that with stochastic gradient descent the θ oscillate around the global minimum, never converging. In most implementations the learning rate is held constant:

![Stochastic Gradient Descent Convergence Example VI](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[16].png)

However, if you want to converge to a minimum you can slowly decrease the learning rate over time, a classic way of doing this is to calculate α as follows:

<i>α = (const1) / (iterationNumber + const2)</i>

But you would need to determine `const1` and `const2`. However, if you tune the parameters well, you can get the algorithm to convert like this:

![Stochastic Gradient Descent Convergence Example VII](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[24].png)
