# Machine Learning Systems

Artificial intelligence and Machine Learning are terms that refer to a technique, a model, or an algorithm that allows computers to learn and think from data. In contrast, machine learning entails much more than just following a series of instructions.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Contents](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#contents)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Contents

- [Recommender Systems](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#recommender-systems)
  - [Problem Formulation](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#problem-formulation)
  - [Content Based Recommendations](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#content-based-recommendations)
    - [How do we learn the parameter vector θ<sub>j</sub>](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#how-do-we-learn-the-parameter-vector-%CE%B8subjsub)
  - [Collaborative Filtering](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#collaborative-filtering)
    - [Formalizing the collaborative filtering problem](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#formalizing-the-collaborative-filtering-problem)
  - [Collaborative Filtering Algorithm](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#collaborative-filtering-algorithm)
    - [Structure](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#structure)
  - [Vectorization: Low Rank Matrix Factorization](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#vectorization-low-rank-matrix-factorization)
  - [Implementational Detail: Mean Normalization](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/1-recommender-systems/#implementational-detail-mean-normalization)
- [Large Scale Machine Learning](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#large-scale-machine-learning)
  - [Learning With Large Datasets](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#learning-with-large-datasets)
  - [Stochastic Gradient Descent](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#stochastic-gradient-descent)
  - [Mini-Batch Gradient Descent](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#mini-batch-gradient-descent)
    - [Mini-Batch Gradient Descent Algorithm](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#mini-batch-gradient-descent-algorithm)
  - [Stochastic Gradient Descent Convergence](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/2-large-scale-machine-learning/#stochastic-gradient-descent-convergence)
- [Online Learning](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/3-online-learning/#online-learning)
  - [Shipping Service Example](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/3-online-learning/#shipping-service-example)
  - [Product Search Example](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/3-online-learning/#product-search-example)
- [Map Reduce and Data Parallelism](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/4-map-reduce-and-data-parallelism/#map-reduce-and-data-parallelism)
  - [Map Reduce](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/4-map-reduce-and-data-parallelism/#map-reduce)
  - [Hadoop](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/4-map-reduce-and-data-parallelism/#hadoop)
- [Application Example: Photo OCR](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#application-example-photo-ocr)
  - [Problem Description and Pipeline](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#problem-description-and-pipeline)
  - [Sliding Windows](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#sliding-windows)
    - [Text Detection](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#text-detection)
    - [Character Segmentation](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#character-segmentation)
    - [Character Classification](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#character-classification)
  - [Getting Lots of Data and Artificial Data](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#getting-lots-of-data-and-artificial-data)
    - [Getting More Data](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#getting-more-data)
  - [Ceiling Analysis: What Part of the Pipeline to Work on Next](https://github.com/rmolinamir/machine-learning-notes/tree/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/#ceiling-analysis-what-part-of-the-pipeline-to-work-on-next)
