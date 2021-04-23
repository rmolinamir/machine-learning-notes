# Recommender Systems

When you buy a product online, most websites automatically recommend other products that you may like. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. In this module, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.

Recommender Systems are an important application of Machine Learning in the tech industry. Many technology companies find recommender systems to be a key factor in their business, such as Amazon, Netflix, Disney, etc. These systems try and recommend new content for you based on passed purchase history and similar historical data sets. They are a substantial part of these companies revenue generation, such as Amazon. It's an absolutely crucial tool in the industry.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem Formulation

Features are important for machine learning, the features you choose will have a big effect on the performance of your learning algorithm. But for some problems, there are algorithms that can try to automatically learn a good set of features for you.

So rather than trying to hand design, there are a few settings where you might be able to have an algorithm learn which features to use, and the recommender systems is just one example of that sort of setting.

For example, you own a company that sells movies, and you let users rate movies using a 0 to 5 star rating system. Let's say that you have five movies, and that you have four users.

![Problem Formulation Example I](https://www.holehouse.org/mlclass/16_Recommender_Systems_files/Image.png)

And:

- n<sub>u</sub> = 4
- n<sub>m</sub> = 5

Where:

- n<sub>u</sub> - Number of users.
- n<sub>m</sub> - Number of movies.
- r(i, j) - 1 if user `j` has rated movie `i`
- y<sub>(i,j)</sub> - ratings given by user `j` to movie `i` (defined only if r(i,j) = 1)

For this example, Alice and Bob gave good ratings to romantic movies, but low scores to action ones. Carol and Dave gave good ratings to action movies but low ratings to romantic movies. With this data, the problem is as follows:

Given r(i,j) and y<sub>(i,j)</sub>, try and predict the missing rating values (`?`) and come up with a learning algorithm that can fill in these missing values.

## Content Based Recommendations
