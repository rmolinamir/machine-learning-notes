# Handling Skewed Data

In the context of evaluation and of error metrics, there is one important case, where it's particularly tricky to come up with an appropriate error metric, or evaluation metric, for your learning algorithm, that case is the case of what's called skewed classes. Consider the problem of cancer classification, where we have features of medical patients and we want to decide whether or not they have cancer. Let's say y equals 1 if the patient has cancer and y equals 0 if they do not. We have trained the progression classifier and let's say we test our classifier on a test set and find that we get 1 percent error.

So only half a percent of the patients that come through our screening process have cancer. In this case, the 1% error no longer looks so impressive. When we're faced with such a skewed classes therefore we would want to come up with a different error metric than accuracy or a different evaluation metric. One such evaluation metric are what's called precision recall.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Error Metrics for Skewed Classes](#error-metrics-for-skewed-classes)
- [Trading Off Precision and Recall](#trading-off-precision-and-recall)
- [Data For Machine Learning](#data-for-machine-learning)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Error Metrics for Skewed Classes

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

## Trading Off Precision and Recall

For many applications, we'll want to somehow control the trade-off between precision and recall.

The F Score, which is also called the F1 Score, is a little bit like taking the average of precision and recall, but it gives a higher weight to the lower value of precision and recall, whichever it is. And so, you see in the numerator here that the F Score takes a product of precision and recall. And so if either precision is 0 or recall is equal to 0, the F Score will be equal to 0. So in that sense, it kind of combines precision and recall, but for the F Score to be large, both precision and recall have to be pretty large. I should say that there are many different possible formulas for combing precision and recall. This F Score formula is really just one out of a much larger number of possibilities, but historically or traditionally this is what people in Machine Learning seem to use:

![Trading Off Precision and Recall F1 Score](blob:https://imgur.com/c9bb13c9-da81-4563-ae58-25305514fe7a)

And the term F Score, it doesn't really mean anything, so don't worry about why it's called F Score or F1 Score. The F Score is used to use precision and recall as an evaluation metric for learning algorithms, more specifically, as a single real number evaluation metric. So you try a range of values of thresholds and evaluate these different thresholds on your cross-validation set and then to pick whatever value of threshold gives you the highest F Score on your cross validation data set. That would be a pretty reasonable way to automatically choose the threshold for your classifier as well.

## Data For Machine Learning

> So, if you have a lot of data and you train a learning algorithm with lot of parameters, that might be a good way to give a high performance learning algorithm, the key test that I often ask myself are first, can a human experts look at the features x and confidently predict the value of y? Because that's sort of a certification that y can be predicted accurately from the features x and second, can we actually get a large training set, and train the learning algorithm with a lot of parameters in the training set and if you can't do both then that's more often give you a very kind performance learning algorithm.

Under certain conditions, getting a lot of data and training on a certain type of learning algorithm, can be a very effective way to get a learning algorithm to do very good performance. This arises often enough that if those conditions hold true for your problem and if you're able to get a lot of data, this could be a very good way to get a very high performance learning algorithm.

Consider a problem of predicting the price of a house from only the size of the house and from no other features. Imagine I tell you that a house is, 500 square feet but I don't give you any other features. I don't tell you that the house is in an expensive part of the city. Or if I don't tell you that the house, the number of rooms in the house, or how nicely furnished the house is, or whether the house is new or old. If I don't tell you anything other than that the house is a 500 square foot house, there's so many other factors that would affect the price of a house other than just the size of a house that if all you know is the size, it's actually very difficult to predict the price accurately.

If we were to go to human expert in this domain, can experts actually confidently predict the value of y? For this first example if we go to an expert realtor and just tell them the size of a house and I tell them what the price is, even an expert in pricing of houses wouldn't be able to tell me knowing only the size doesn't provide enough information to predict the price of the house.

But, when having a lot of data could help? Suppose the features have enough information to predict the value of y. And let's suppose we use a learning algorithm with a large number of parameters, chances are, if we run these algorithms on the data sets, it will be able to fit the training set well, and so hopefully the training error will be slow. Let's say, we use a massive, massive training set, then hopefully even though we have a lot of parameters the training set is much larger than the number of parameters, in thes case the algorithm will be unlikely to overfit which means is that the training error will hopefully be close to the test error, which means the test error will be small.
