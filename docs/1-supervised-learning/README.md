# Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "Regression" and "Classification" problems.

In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.

In a classification problem, we are instead trying to predict results in a discrete output.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Example 1](#example-1)
- [Example 2](#example-2)
- [Contents](#contents)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Example 1

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "Sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

## Example 2

- Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
- Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "Regression" and "Classification" problems.

In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.

In a classification problem, we are instead trying to predict results in a discrete output.

## Contents

- [Cost Function](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/1-cost-function/#cost-function)
  - [Cost Function Intuition 1](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/1-cost-function/#cost-function-intuition-1)
  - [Cost Function Intuition 2](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/1-cost-function/#cost-function-intuition-2)
- [Gradient Descent](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/2-gradient-descent/#gradient-descent)
  - [Gradient Descent Algorithm](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/2-gradient-descent/#gradient-descent-algorithm)
  - [Gradient Descent Intuition](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/2-gradient-descent/#gradient-descent-intuition)
  - [Gradient Descent For Linear Regression](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/2-gradient-descent/#gradient-descent-for-linear-regression)
- [Linear Regression with Multiple Variables (or Features)](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#linear-regression-with-multiple-variables-or-features)
  - [Gradient Descent in Practice I: Feature Scaling](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#gradient-descent-in-practice-i-feature-scaling)
  - [Gradient Descent in Practice I: Learning Rate](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#gradient-descent-in-practice-i-learning-rate)
  - [Features and Polynomial Regression](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#features-and-polynomial-regression)
  - [Normal Equation](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#normal-equation)
  - [Normal Equation Noninvertibility](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#normal-equation-noninvertibility)
  - [Vectorized Implementations](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#vectorized-implementations)
    - [Sources](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#sources)
    - [Hypothesis Vectorization](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#hypothesis-vectorization)
    - [Cost Function Vectorization](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#cost-function-vectorization)
    - [Cost Function Derivation Vectorization](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#cost-function-derivation-vectorization)
    - ["Side" Mentions](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/3-linear-regression-with-multiple-variables/#side-mentions)
- [Classification](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#classification)
  - [Hypothesis Representation](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#hypothesis-representation)
  - [Logistic Regression Cost Function](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#logistic-regression-cost-function)
  - [Advanced Optimization Algorithms for Logistic Regresion](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#advanced-optimization-algorithms-for-logistic-regresion)
  - [Multiclass Classification: One-vs-all](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#multiclass-classification-one-vs-all)
  - [Overfitting](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#overfitting)
  - [Regularization Cost Function](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#regularization-cost-function)
  - [Regularized Linear Regression](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#regularized-linear-regression)
    - [Regularized Linear Regression Gradient Descent](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#regularized-linear-regression-gradient-descent)
    - [Regularized Linear Regression Normal Equation](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#regularized-linear-regression-normal-equation)
  - [Regularized Logistic Regression](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/4-classification/#regularized-logistic-regression)
- [Neural Networks](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#neural-networks)
  - [Neural Networks Model Representation](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#neural-networks-model-representation)
  - [Neural Networks Cost Function](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#neural-networks-cost-function)
  - [Backpropagation Algorithm](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#backpropagation-algorithm)
  - [Gradient Checking](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#gradient-checking)
  - [Random Initialization (for Theta parameters)](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#random-initialization-for-theta-parameters)
  - [Putting it Together](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/5-neural-networks/#putting-it-together)
- [Diagnostics](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#diagnostics)
  - [Diagnosing Neural Networks](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#diagnosing-neural-networks)
  - [Evaluating a Hypothesis](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#evaluating-a-hypothesis)
  - [Model Selection and Train/Validation/Test Sets](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#model-selection-and-trainvalidationtest-sets)
  - [Diagnosing Bias vs. Variance](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#diagnosing-bias-vs-variance)
  - [Regularization and Bias/Variance](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#regularization-and-biasvariance)
  - [Learning Curves](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#learning-curves)
  - [Neural Networks and Overfitting](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/6-diagnostics/#neural-networks-and-overfitting)
- [Prioritizing What to Work On](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/7-prioritizing-what-to-work-on/#prioritizing-what-to-work-on)
  - [Error Analysis](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/7-prioritizing-what-to-work-on/#error-analysis)
- [Handling Skewed Data](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/8-handling-skewed-data/#handling-skewed-data)
  - [Error Metrics for Skewed Classes](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/8-handling-skewed-data/#error-metrics-for-skewed-classes)
  - [Trading Off Precision and Recall](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/8-handling-skewed-data/#trading-off-precision-and-recall)
  - [Data For Machine Learning](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/8-handling-skewed-data/#data-for-machine-learning)
- [Support Vector Machines](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#support-vector-machines)
  - [Optimization Objective](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#optimization-objective)
  - [Large Margin Intuition](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#large-margin-intuition)
  - [Mathematics Behind Large Margin Classification](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#mathematics-behind-large-margin-classification)
  - [Kernels I](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#kernels-i)
  - [Kernels II](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#kernels-ii)
    - [Choosing the landmarks for Gaussian kernels](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#choosing-the-landmarks-for-gaussian-kernels)
    - [Choosing parameters C and σ<sup>2</sup> for Gaussian kernels](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#choosing-parameters-c-and-σsup2sup-for-gaussian-kernels)
  - [Using An SVM](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#using-an-svm)
    - [Choosing a kernel](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#choosing-a-kernel)
      - [Gaussian kernel](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#gaussian-kernel)
      - [Linear kernel](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#linear-kernel)
      - [Polynomial kernel](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#polynomial-kernel)
      - [String kernel](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#string-kernel)
    - [Multi-class classification for SVM](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#multi-class-classification-for-svm)
    - [When to use Logistic Regression or SVM?](https://github.com/rmolinamir/machine-learning-introduction/tree/main/docs/1-supervised-learning/9-support-vector-machines/#when-to-use-logistic-regression-or-svm)
