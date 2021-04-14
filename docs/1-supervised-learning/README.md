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

- [Cost Function](#cost-function)
  - [Cost Function Intuition 1](#cost-function-intuition-1)
  - [Cost Function Intuition 2](#cost-function-intuition-2)
- [Gradient Descent](#gradient-descent)
  - [Gradient Descent Algorithm](#gradient-descent-algorithm)
  - [Gradient Descent Intuition](#gradient-descent-intuition)
  - [Gradient Descent For Linear Regression](#gradient-descent-for-linear-regression)
- [Linear Regression with Multiple Variables (or Features)](#linear-regression-with-multiple-variables-or-features)
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
- [Classification](#classification)
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
- [Neural Networks](#neural-networks)
  - [Neural Networks Model Representation](#neural-networks-model-representation)
  - [Neural Networks Cost Function](#neural-networks-cost-function)
  - [Backpropagation Algorithm](#backpropagation-algorithm)
  - [Gradient Checking](#gradient-checking)
  - [Random Initialization (for Theta parameters)](#random-initialization-for-theta-parameters)
  - [Putting it Together](#putting-it-together)
- [Diagnostics](#diagnostics)
  - [Diagnosing Neural Networks](#diagnosing-neural-networks)
  - [Evaluating a Hypothesis](#evaluating-a-hypothesis)
  - [Model Selection and Train/Validation/Test Sets](#model-selection-and-trainvalidationtest-sets)
  - [Diagnosing Bias vs. Variance](#diagnosing-bias-vs-variance)
  - [Regularization and Bias/Variance](#regularization-and-biasvariance)
  - [Learning Curves](#learning-curves)
  - [Neural Networks and Overfitting](#neural-networks-and-overfitting)
- [Prioritizing What to Work On](#prioritizing-what-to-work-on)
  - [Error Analysis](#error-analysis)
- [Handling Skewed Data](#handling-skewed-data)
  - [Error Metrics for Skewed Classes](#error-metrics-for-skewed-classes)
  - [Trading Off Precision and Recall](#trading-off-precision-and-recall)
  - [Data For Machine Learning](#data-for-machine-learning)
- [Support Vector Machines](#support-vector-machines)
  - [Optimization Objective](#optimization-objective)
  - [Large Margin Intuition](#large-margin-intuition)
  - [Mathematics Behind Large Margin Classification](#mathematics-behind-large-margin-classification)
  - [Kernels I](#kernels-i)
  - [Kernels II](#kernels-ii)
    - [Choosing the landmarks for Gaussian kernels](#choosing-the-landmarks-for-gaussian-kernels)
    - [Choosing parameters C and σ<sup>2</sup> for Gaussian kernels](#choosing-parameters-c-and-σsup2sup-for-gaussian-kernels)
  - [Using An SVM](#using-an-svm)
    - [Choosing a kernel](#choosing-a-kernel)
      - [Gaussian kernel](#gaussian-kernel)
      - [Linear kernel](#linear-kernel)
      - [Polynomial kernel](#polynomial-kernel)
      - [String kernel](#string-kernel)
    - [Multi-class classification for SVM](#multi-class-classification-for-svm)
    - [When to use Logistic Regression or SVM?](#when-to-use-logistic-regression-or-svm)
