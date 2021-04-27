# Recommender Systems

When you buy a product online, most websites automatically recommend other products that you may like. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. In this module, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.

Recommender Systems are an important application of Machine Learning in the tech industry. Many technology companies find recommender systems to be a key factor in their business, such as Amazon, Netflix, Disney, etc. These systems try and recommend new content for you based on passed purchase history and similar historical data sets. They are a substantial part of these companies revenue generation, such as Amazon. It's an absolutely crucial tool in the industry.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Problem Formulation](#problem-formulation)
- [Content Based Recommendations](#content-based-recommendations)
  - [How do we learn the parameter vector θ<sub>j</sub>](#how-do-we-learn-the-parameter-vector-%CE%B8subjsub)
- [Collaborative Filtering](#collaborative-filtering)
  - [Formalizing the collaborative filtering problem](#formalizing-the-collaborative-filtering-problem)
- [Collaborative Filtering Algorithm](#collaborative-filtering-algorithm)
  - [Structure](#structure)
- [Vectorization: Low Rank Matrix Factorization](#vectorization-low-rank-matrix-factorization)
- [Implementational Detail: Mean Normalization](#implementational-detail-mean-normalization)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem Formulation

Features are important for machine learning, the features you choose will have a big effect on the performance of your learning algorithm. But for some problems, there are algorithms that can try to automatically learn a good set of features for you.

So rather than trying to hand design, there are a few settings where you might be able to have an algorithm learn which features to use, and the recommender systems is just one example of that sort of setting.

For example, you own a company that sells movies, and you let users rate movies using a 0 to 5 star rating system. Let's say that you have five movies, and that you have four users.

![Problem Formulation Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Problem-Formulation%20Example%20I.png)

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

How do we predict user ratings? With the example above, for each movie we have features which measure characteristics of a movie, such as romance, action, etc. In this example, we have:

- x<sup>1</sup>: measuring romance.
- x<sup>2</sup>: measuring action.

![Content Based Recommendations Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20I.png)

If we have features like these, each movie can be recommended by a feature vector. We also add the intercept term x<sup>0</sup> = 1 just like in linear regressions. To be consistent with our notation, `n` is going to be the number of features NOT counting the x<sup>0</sup> term, so n = 2, and each user rating will be treated as a separate linear regression problem.

For each user `j` we could learn a parameter vector `θ`, then predict how will user `j` rate movie `i` using:

- (θ<sup>j</sup>)<sup>T</sup>x<sup>i</sup> = r

Where `r` are the star ratings, and it's the product of the parameter vector and the movie features.

For example, let's take user 1 (Alice) and see what she makes of the modern classic Cute Puppies of Love (CPOL). We have some parameter vector (θ<sup>1</sup>) associated with Alice (we'll explain later how we derived these values):

![Content Based Recommendations Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20II.png)

CPOL has a feature vector (x<sup>3</sup>) associated with it:

![Content Based Recommendations Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20III.png)

Applying the equation used to compute the user rating, our prediction will be equal to:

(θ<sup>j</sup>)<sup>T</sup>x<sup>i</sup> = (0 \* 1) + (5 \* 0.99) + (0 * 0) = 4.95

Which may seem like a reasonable value. All we're doing here is applying a linear regression method for each user, so we determine a future rating based on their interest in romance and action features based on previous user ratings.

We should also add one final piece of notation m<sup>j</sup>, which is the number of movies rated by user `j`.

### How do we learn the parameter vector θ<sub>j</sub>

This is just like the least squares regressions in linear regression, where we want to choose θ<sub>j</sub> (the parameter vector theta `j`) to minimize this type of squared error term.

![Content Based Recommendations Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20IV.png)

To do this:

- Sum over all values of `i` (all movies the user has used) when r(i,j) = 1 (i.e. all the movies that the user has rated).
- We can also add a regularization term to make our equation look as follows:
    ![Content Based Recommendations Example V](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20V.png)

The regularization term goes from k = 1 to m, so (θ<sub>j</sub>) ends up being a n + 1 feature vector, just like in linear regression we won't regularize over the bias terms (θ<sub>0</sub>). If you do this, you will get a reasonable value.

To make this a little bit clearer you can get rid of the m<sup>j</sup> term (because it's just a constant it shouldn't make any difference to minimization similar to the SVM algorithm).

So to learn θ<sup>j</sup>:

![Content Based Recommendations Example VI](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20VI.png)

But for our recommender system we want to learn parameters for *all* users, so we add an extra summation term to this which means we determine the minimum θ<sup>j</sup> value for every user:

![Content Based Recommendations Example VII](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20VII.png)

When you do this as a function of each θ<sup>j</sup> parameter vector you get the parameters for each user. This is our optimization objective: J(θ<sup>1</sup>, ..., θ<sup>n<sub>u</sub></sup>).

In order to do the minimization we have the following gradients:

![Content Based Recommendations Example VIII](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20VIII.png)

But because we're not regularizing the bias term (our regularization term here regularizes only the values of theta θ<sup>j</sup> for k not equal to 0, so we don't regularize θ<sup>0</sup>), we can write it slightly differently to our previous gradient descent implementations:

![Content Based Recommendations Example IX](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Content-Based%20Recommendations%20Example%20IX.png)

This approach is called content-based approach because we assume we have features regarding the content which will help us identify things that make them appealing to a user. However, often such features are not available.

## Collaborative Filtering

The collaborative filtering algorithm has a very interesting property, it performs feature learning. In other words, it can learn for itself what features it needs to use.

Recall our original data set above for our five movies and four users, let's assume someone had calculated the "romance" and "action" amounts of the movies, this has some disadvantages:

- This can be very hard to do in reality.
- We often want more features than just two.

Let's change the problem and pretend we have a data set where we don't know any of the features associated with the movies.

![Collaborative Filtering Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Example%20I.png)

Now let's make a different assumption, we've polled each user and found out how much each user likes romantic and action movies and generated the following parameter set:

![Collaborative Filtering Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Example%20II.png)

- Alice and Bob like romance but dislike action.
- Carol and Dave like action but dislike romance.

If we can get these parameters from the users we can infer the missing values from our table, let's look at "Love at Last":

- Alice and Bob loved it.
- Carol and Dave dislike it.

We know from the feature vectors that Alice and Bob love romantic movies, while Carol and Dave dislike them. Based on Alice and Bob liking "Love at Last" and Carol and Dave disliking it, we may be able to (correctly) conclude that "Love at Last" is a romantic movie.

This is a bit of a simplification in terms of the maths, but what we're really asking is "What feature vector should x<sup>1</sup> be so that":

- (θ<sup>1</sup>)<sup>T</sup> x<sup>1</sup> is about 5.
- (θ<sup>2</sup>)<sup>T</sup> x<sup>1</sup> is about 5.
- (θ<sup>3</sup>)<sup>T</sup> x<sup>1</sup> is about 0.
- (θ<sup>4</sup>)<sup>T</sup> x<sup>1</sup> is about 0.

From this we can estimate that x<sup>1</sup> is:

![Collaborative Filtering Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Example%20III.png)

Using that same approach we should then be able to determine the remaining feature vectors for the other movies.

### Formalizing the collaborative filtering problem

We can more formally describe the approach as follows:

- Given θ<sup>1</sup>, ..., θ<sup>n<sub>u</sub></sup>, (i.e. given the parameter vectors for each users' preferences),
- We must minimize an optimization function which tries to identify the best parameter vector associated with a movie:
  ![Formalizing the collaborative filtering problem Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Formalizing-the%20collaborative%20filtering%20problem%20Example%20I.png)
- Assuming over all the indices `j` for where we have data for movie `i`, we're minimizing this squared error.
- Like before, the above equation gives us a way to learn the features for one movie.
- We want to learn all the features for all the movies - so we need an additional summation term.

In the previous recommendation system we have users preferences based on their ratings, so now we can therefore determine a movie's features. But this is a bit of a chicken & egg problem. What you can do is:

- Randomly guess values for θ.
- Use collaborative filtering to generate x.
- Use content based recommendation to improve θ.
- Use that to improve x.
- Repeat until convergence.

This causes the collaborative filtering algorithm to converge on a reasonable set of parameters. We call it collaborative filtering because in this example the users are collaborating together to help the algorithm learn better features and help the system and the other users.

## Collaborative Filtering Algorithm

Here we combine the ideas from before to build a collaborative filtering algorithm, our starting point is as follows:

1. If we're given the movie's features we can use that to work out the users' preferences:
    ![Collaborative Filtering Algorithm Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Algorithm%20Example%20I.png)
    - If we're given the users' preferences we can use them to work out the movie's features:
    ![Collaborative Filtering Algorithm Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Algorithm%20Example%20II.png)
2. One thing you could do is:
   - Randomly initialize the parameters θ.
   - Compute back and forward between θ and x values until convergence.

But there's a more efficient algorithm which can solve θ and x simultaneously by defining a new optimization objective which is a function of θ and x:

![Collaborative Filtering Algorithm Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Algorithm%20Example%20III.png)

The squared error term is the same as the squared error term in the two individual objectives above:

![Collaborative Filtering Algorithm Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Collaborative-Filtering%20Algorithm%20Example%20IV.png)

- It's summing over every movie rated by every users, and the ":" character means, "for which".
- so it reads as "Sum over all pairs (i,j) for which r(i,j) is equal to 1".

The regularization terms are simply added to the end from the original two optimization functions.

This newly defined function has the property that if you held x constant and only solved θ then you compute the optimum values of θ and x that minimize the cost. In order to come up with just one optimization function we treat this function as a function of both movie features x and user parameters θ, the only difference between this in the back-and-forward approach is that we minimize with respect to both x and θ simultaneously.

When we're learning the features this way:

- Previously had a convention that we have an x<sup>0</sup> = 1 term, but with this approach there is no need for a bias term.
  - So now our vectors (both x and θ) are n-dimensional (not n + 1, but n).
- This is because we are now learning all the features so if the system needs a feature always equal to 1 then the algorithm can learn one.

### Structure

1. Initialize θ<sup>1</sup>, ..., θ<sup>n<sub>u</sub></sup>, and x<sup>1</sup>, ..., x<sup>n<sub>m</sub></sup> to small random and different values like in neural networks.
2. Minimize cost function J(θ<sup>1</sup>, ..., θ<sup>n<sub>u</sub></sup>, and x<sup>1</sup>, ..., x<sup>n<sub>m</sub></sup>) using gradient descent:
   - We find that the update rules look like this:
     ![Feature Gradient](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Feature-Gradient.png)
     ![Parameter Gradient](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Parameter-Gradient.png)
   - Where:
     - Where the top term is the partial derivative of the cost function with respect to x<sub>k<sup>(i)</sup></sub> while the bottom is the partial derivative of the cost function with respect to θ<sub>k<sup>(i)</sup></sub>.
     - We regularize every parameter of the cost function because there is no bias term, so there is no special case update rule.
3. Having minimized the values, given a user (user `j`) with parameters θ and movie (movie `i`) with learned features x, we predict a start rating of (θ<sup>j</sup>)<sup>T</sup>x<sup>i</sup>. This is the collaborative filtering algorithm, which should give pretty good predictions for how users like new movies.

## Vectorization: Low Rank Matrix Factorization

Having looked at the collaborative filtering algorithm, how can we improve the algorithm? Given one product, can we determine other relevant products? We start by working out another way of writing out our predictions. So take all ratings by all users in our example above and group into a matrix Y:

![Vectorization: Low Rank Matrix Factorization Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Vectorization-Low%20Rank%20Matrix%20Factorization%20Example%20I.png)

Where 5 movies and 4 users are composed into a [5 x 4] matrix.

Given [Y] there's another way of writing out all the predicted ratings:

![Low Rank Matrix Factorization Matrix Y](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Low-Rank%20Matrix%20Factorization%20Matrix%20Y.png)

With this matrix of predictive ratings we determine the (i,j) entry for EVERY movie.

We can define another matrix X (just like matrix we had for linear regression), take all the features for each movie and stack them in rows (think of each movie as one example):

![Low Rank Matrix Factorization Matrix X](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Low-Rank%20Matrix%20Factorization%20Matrix%20X.png)

And do the same for matrix Θ, taking each parameter per user vector and stack in rows:

![Low Rank Matrix Factorization Matrix Θ](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Low-Rank%20Matrix%20Factorization%20Matrix%20%CE%98.png)

Given our new matrices X and θ we can have a vectorized way of computing the prediction range matrix by doing X \* θ<sup>T</sup>. We can give this algorithm another name - the low rank matrix factorization. This comes from the property that the X \* θ<sup>T</sup> calculation has a property in linear algebra where we create a low rank matrix.

Finally, having run the collaborative filtering algorithm, we can use the learned features to find related movies.

When you learn a set of features you don't know what the features will be. Let's say you identify the features which define a movie, for example, we learn the following features:

- x<sub>1</sub> - Romance
- x<sub>2</sub> - Action
- x<sub>3</sub> - Comedy
- x<sub>4</sub> - ...

So we have n features all together. After you've learned these features, it's often very hard to come in and apply a human understandable metric to what those features are.

Now that you have learned these feature vectors, this gives us a very convenient way to measure how similar two movies are. Our features allow a good way to measure movie similarity.

If we have two movies x<sup>i</sup> and x<sup>j</sup>, we want to minimize ||x<sup>i</sup> - x<sup>j</sup>||, i.e. the distance between those two movies. This provides a good indicator of how similar two movies are, and this could give you a few different movies to recommend to your user.

## Implementational Detail: Mean Normalization

We have one final implementation detail to make the algorithm work a bit better. To show why we might need mean normalization, let's consider an example where there's a user who hasn't rated any movies:

![Implementational Detail: Mean Normalization Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20I.png)

Lets say that the algorithm does the following for this user:

- Say n = 2.
- We now have to learn θ<sup>5</sup> (which is an n-dimensional vector).

Looking in the first term of the optimization objective:

- There are no movies for which r(i,j) = 1.
- So this term places no role in determining θ<sup>5</sup>.
- So we're just minimizing the final regularization term.

![Implementational Detail: Mean Normalization Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20II.png)

If the goal is to minimize this term then:

![Implementational Detail: Mean Normalization Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20III.png)

This makes sense because if there's no data to pull the values away from 0, we predict ANY movie to be zero. But presumably Eve doesn't hate all movies, so if we're doing this we can't recommend any movies to her either. **Mean normalization should let us fix this problem**. How does mean normalization work?

Group all our ratings into matrix Y as before. We now have a column of indetermined values (?) which corresponds to Eve's ratings, and then we compute the average rating each movie obtained and store in an n<sub>m</sub> dimensional vector.

![Implementational Detail: Mean Normalization Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20IV.png)

![Implementational Detail: Mean Normalization Example V](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20V.png)

If we look at all the movie ratings in Y we can subtract off the mean rating:

![Implementational Detail: Mean Normalization Example VI](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20VI.png)

This means we normalize each movie to have an average rating of 0. Now, we take the new set of ratings and use it with the collaborative filtering algorithm. We learn θ<sup>j</sup> and x<sup>i</sup> from the mean normalized ratings, so for our prediction of user `j` on movie `i`, predict:

(θ<sup>j</sup>)<sup>T</sup>x<sup>i</sup> + μ<sub>i</sub>

- Where these vectors are the mean normalized values.
- We have to add μ because we removed it from our θ values.

But for user 5 the same argument applies:

![Implementational Detail: Mean Normalization Example VII](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/1-recommender-systems/images/Implementational-Detail%20Mean%20Normalization%20Example%20VII.png)

On any movie `i` we're going to predict (θ<sup>5</sup>)<sup>T</sup>x<sup>5</sup> + μ<sub>5</sub>, where (θ<sup>5</sup>)<sup>T</sup>x<sup>5</sup> = to 0 still, but we then add the mean (μ<sub>i</sub>) which means Eve has an average rating assigned to each movie.

This makes sense, because if Eve hasn't rated any movies, we predict the average rating of the movies based on everyone's ratings.

As an aside - we spoke here about mean normalization for users with no ratings.

- If you have some movies with no ratings you can also play with versions of the algorithm where you normalize the columns.
- BUT this is probably less relevant, you probably shouldn't recommend an unrated movie.

To summarize, this shows how you do mean normalization preprocessing to allow your system to deal with users who have not yet made any ratings. This means we recommend a user that we know little about the best average rated movies.
