# Map Reduce and Data Parallelism

Some machine learning problems are just too big to run on one machine, sometimes maybe you have so much data you just don't ever want to run all that data through a single computer, no matter what algorithm you would use on that computer.

Let's say we want to fit a linear regression model or a logistic regression model or some such, and let's start again with batch gradient descent, and to keep the writing on this slide tractable, I'm going to assume throughout that we have m equals 400 examples. Of course, by our standards, in terms of large scale machine learning, you know m might be pretty small and so, this might be more commonly applied to problems where you have maybe closer to 400 million examples, but just to make this simlper, I'm going to pretend we have 400 examples.

In that case, the batch gradient descent has to process these 400 examples, and if m is large, then this is a computationally expensive step.

The Map Reduce idea does is the following, and I should say the map reduce idea is due to two researchers, Jeff Dean and Sanjay Gimawat.

- Jeff Dean, by the way, is one of the most legendary engineers in all of Silicon Valley and he kind of built a large fraction of the architectural infrastructure that all of Google runs on today.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

- [Map Reduce](#map-reduce)
- [Hadoop](#hadoop)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Map Reduce

Let's say I have some training set, and we start by splitting the training sets into different subsets. In this example, we split the training set into 4 pieces.

- Machine 1: use (x<sup>1</sup>, y<sup>1</sup>), ..., (x<sup>100</sup>, y<sup>100</sup>).
  - Uses first quarter of training set
  - Just does the summation for the first 100
  ![Map Reduce Example I](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[27].png)

- Similarly for the rest of the machines:
  ![Map Reduce Example II](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[28].png)

Each of these `temp` values are partial sums of the gradient. So now we have these four values, and each machine does 1/4 of the work, we send these values to a centralized master server, put them back together, and update θ using:

![Map Reduce Example III](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[29].png)

This equation is doing the same as our original batch gradient descent algorithm. More generally map reduce uses the following scheme (e.g. where you split into 4):

![Map Reduce Example IV](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[30].png)

The bulk of the work in gradient descent is the summation, but now, because each of the computers does a quarter of the work at the same time, you get a 4x speedup. In reality, because of network latency and combining the results, it's slightly less than 4x, but close.

The important thing to ask is: "Can the algorithm be expressed as computing sums of functions of the training set?" Turns out, many algorithms can.

Another example:

Using an advanced optimization algorithm with logistic regression:

![Map Reduce Example V](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[31].png)

Now we need to calculate cost function, so we split the training set into `x` machines, and these `x` machines compute the sum of the value over `1/x`<sup>th</sup> of the data.

![Map Reduce Example VI](https://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning_files/Image%20[32].png)

These terms are also a sum over the training set, we're using the same approach. We send the resulting `temp` values to a central server to deal with combining everything.

More broadly, by taking algorithms which compute sums you can scale them to very large datasets through parallelization. Parallelization can come from:

- Multiple machines.
- Multiple CPUs.
- Multiple cores in each CPU.

So even on a single computer, we can implement parallelization. The advantage of thinking about Map Reduce here is that you don't need to worry about network issues as long as the implementation is aptly scaled.

Finally, depending on implementation detail, certain numerical linear algebra libraries can automatically parallelize your calculations across multiple cores. So, if this is the case and you have a good vectorization implementation you can ignore worrying about local parallelization and the local libraries sort optimization out for you.

## Hadoop

Hadoop is a good open source Map Reduce implementation. It represents a top-level Apache project developed by a global community of developers written in Java. Yahoo has been the biggest contributor and pushed a lot early on, now it's receiving a support now from Cloudera.
