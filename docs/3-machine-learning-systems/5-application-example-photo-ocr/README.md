# Application Example: Photo OCR

We will walk through a complex, end-to-end application of machine learning, to the application of Photo OCR. Identifying and recognizing objects, words, and digits in an image is a challenging task. We discuss how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

- [Problem Description and Pipeline](#problem-description-and-pipeline)
- [Sliding Windows](#sliding-windows)
  - [Text Detection](#text-detection)
  - [Character Segmentation](#character-segmentation)
  - [Character Classification](#character-classification)
- [Getting Lots of Data and Artificial Data](#getting-lots-of-data-and-artificial-data)
  - [Getting More Data](#getting-more-data)
- [Ceiling Analysis: What Part of the Pipeline to Work on Next](#ceiling-analysis-what-part-of-the-pipeline-to-work-on-next)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem Description and Pipeline

With growth of digital photography, there are lots of digital pictures nowadays. One idea which has interested many people is getting computers to understand those photos, and the photo OCR problem is getting computers to read text in an image. Possible applications for this would include:

- Make searching easier (e.g. searching for photos based on words in them).
- Car navigation.

OCR of documents is a comparatively easy problem because letters are more standardized, but from photos is much harder.

This is a real case study focused around photo OCR (optical character recognition). We will:

1. Look at how a complex system can be put together.
2. The idea of a machine learning pipeline:
   - What to do next.
   - How to do it.
3. Some more interesting ideas:
   - Applying machine learning to tangible problems.
   - Artificial data synthesis.

Pipelines are common in machine learning. They are separate modules which may each be a machine learning component or data processing component. If you're designing a machine learning system, pipeline design is one of the most important questions.

Performance of pipeline and each module often has a big impact on the overall performance a problem and you would often have different engineers working on each module, this offers a natural way to divide up the workload.

Our Photo OCR pipeline looks like this:

1. Look through images and find text.
2. Do character segmentation.
3. Do character classification.
4. Optionally, some may do spell check after this too (we're not focussing on such systems though).

![Problem Description and Pipeline](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Problem-Description%20and%20Pipeline.png)

## Sliding Windows

As mentioned, stage 1 is text detection. How do the individual models work? We will focus on a sliding windows classifier.

This is an unusual problem in computer vision, different rectangles (which surround text) may have different aspect ratios, text may be short (few words) or long (many words), with tall or short font, the text might be straight on or slanted, etc.

![Sliding Windows Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Sliding-Windows%20Example%20I.png)

Let's start with a simpler example of pedestrian detection where we want to take an image and find pedestrians in the image:

![Sliding Windows Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Sliding-Windows%20Example%20II.png)

This is a slightly simpler problem because the aspect ration remains pretty constant, building our detection system. With a 82x36 aspect ratio being a typical aspect ratio for a standing human, we collect a training set of positive and negative examples:

![Sliding Windows Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Sliding-Windows%20Example%20III.png)

We could have 1000 - 10000 training examples and train a neural network to take an image and classify that image as pedestrian or not, this gives you a way to train your system. But now we have a new image below, how do we find pedestrians in it?

![Sliding Windows Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Sliding-Windows%20Example%20IV.png)

- We start by taking a rectangular 82x36 patch in the image.
  - We run this patch (green rectangle) through the classifier, and hopefully in this example it will return y = 0.
- Next, slide the rectangle over to the right a little bit and re-run the classifier, then slide again over and over until we slide through the image completely.
  - The amount you slide each rectangle over is a parameter called the step-size or stride.
  - Could use 1 pixel (at best), but it would be computationally expensive, more commonly it's 5-8 pixels.
- We keep stepping the patch along all the way to the right. Eventually we get to the end then move back to the left hand side but step down a bit too.
- Repeat until you've covered the whole image
- Now, we initially started with quite a small rectangle.
  - So now we can take a larger image patch (of the same aspect ratio).
  - Each time we process the image patch, we're resizing the larger patch to a smaller image, then running that smaller image through the classifier.
- Hopefully, by changing the patch size and rastering repeatedly across the image, you eventually recognize all the pedestrians in the picture.

![Sliding Windows Example V](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Sliding-Windows%20Example%20V.png)

### Text Detection

For text detection, like pedestrian detection, we generate a labeled training set with positive examples (some kind of text) and negative examples (not text).

![Text Detection Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Text-Detection%20Example%20I.png)

Having trained the classifier we apply it to an image. So, we run a sliding window classifier at a fixed rectangle size. If you do that end up with something like this:

![Text Detection Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Text-Detection%20Example%20II.png)

White region show where text detection system thinks text is. Different shades of gray correspond to probability associated with how sure the classifier is the section contains text:

- Black means no text.
- White means text.

For text detection, we want to draw rectangles around all the regions where there is text in the image, so we take the classifier output and apply an expansion algorithm which takes each of the white regions and expands them. How do we implement this?

Say, for every pixel, is it within some distance of a white pixel? If yes then colour it white:

![Text Detection Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Text-Detection%20Example%20III.png)

We look at the connected white regions in the image above and draw rectangles around those which make sense as text (i.e. tall thin boxes don't make sense).

![Text Detection Example IV](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Text-Detection%20Example%20IV.png)

This example misses a piece of text on the door because the aspect ratio is wrong, it's hard to read.

### Character Segmentation

Stage two of the pipeline is character segmentation. We will use a supervised learning algorithm. Look in a defined image patch and decide, is there a split between two characters? So, for example, our first training data item below looks like there is such a split. Similarly, the negative examples are either empty or hold full characters (i.e. there is no split between two characters in the middle).

![Character Segmentation Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Character-Segmentation%20Example%20I.png)

We train a classifier to try and classify between positive and negative examples, then we run that classifier on the regions detected as containing text in the previous section. We use a 1-dimensional sliding window (horizontal movement only) to move along text regions to determine if each window snapshot looks like the split between two characters.

- If yes insert a split.
- If not move on.

So we have something that looks like this:

![Character Segmentation Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Character-Segmentation%20Example%20II.png)

### Character Classification

Finally, the last stage of the pipeline is the character classification component. This would be a standard OCR, where you apply standard supervised learning which takes an input and identify which character we decide it is. This is a multi-class characterization problem similar to the digits OCR neural network algorithm found in the neural networks section.

## Getting Lots of Data and Artificial Data

We've seen over and over that one of the most reliable ways to get a high performance machine learning system is to take a low bias algorithm and train on a massive data set. Is there any way can we get much data from unconventional channels?

In machine learning, artificial data synthesis doesn't apply to every problem, but if it applies to your problem then it can be a great way to generate loads of data.

There are two main principles:

1. Creating data from scratch
2. If we already have a small labeled training set, we can we amplify it into a larger training set:

For example, if we go and collect a large labeled data set, it will look like this.

- Our goal is to take an image patch and have the system recognize the character.
- We treat the images as gray-scale (makes it a bit easer since color has no much impact).

![Getting Lots of Data and Artificial Data Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Getting-Lots%20of%20Data%20and%20Artificial%20Data%20Example%20I.png)

How can we amplify this? Modern computers often have a big font library. If you go to websites, you can get huge free font libraries. For more training data, we can take characters from different fonts, and paste these characters again on top of random backgrounds.

After some work, can build a synthetic training set like this:

![Getting Lots of Data and Artificial Data Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Getting-Lots%20of%20Data%20and%20Artificial%20Data%20Example%20II.png)

- Random background.
- Apply some blurring/distortion filters.
- Takes thought and work to make it look realistic.
- This is an example of creating new data from scratch.

It's important to note that **if you do a sloppy job this won't help**.

The above is an example of creating new data from scratch, but there are other ways to introduce distortion into existing data, e.g. take a character and warp it.

![Getting Lots of Data and Artificial Data Example III](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Getting-Lots%20of%20Data%20and%20Artificial%20Data%20Example%20III.png)

This way we generate 16 new examples per example. It allows you to amplify existing training set. But, it also takes thought and insight in terms of deciding how to amplify the data set.

Another example is speech recognition. Learn from an audio clip what were the words said. To do this we have a labeled training example and introduce audio distortions into the examples. So from just one example, we can create lots of new ones. When introducing distortion, they should be reasonable relative to the issues your classifier may encounter.

### Getting More Data

Before creating new data, make sure you have a low bias classifier, plot a learning curve. If it's not a low bias classifier then increasing the number of features might help, consider creating a large artificial training set.

A very important question to ask is: How much work would it be to get 10 times the data as we currently have? Often the answer is, "Not that hard", this is often a huge way to improve an algorithm. Good question to ask yourself or ask the team.

How many minutes/hours does it take to get a certain number of examples? For example, say we have 1000 training set examples:

- It takes 10 seconds to label an example.
- If we want 10000 examples, we need another 100000 seconds.
- It results in just a few days of work (~28 hours).

Crowd sourcing is also a good way to get data but there are risk or reliability issues, but the cost is usually economical. An example of crowdsourcing would be Amazon's Mechanical Turks.

## Ceiling Analysis: What Part of the Pipeline to Work on Next

We have epeatedly said one of the most valuable resources is developer time. We should always pick the right thing for you and your team to work on and aoid spending a lot of time to realize the work was pointless in terms of enhancing performance.

In our photo OCR pipeline we have three modules:

- Each one could have a small team on it.
- Where should you allocate resources?.
- It's good to have a single real number as an evaluation metric.
  - So, character accuracy for this example.

Say, we find that our test set has 72% accuracy. Let's perform a ceiling analysis on our pipeline. We go to the first module and mess around with the test set, we manually tell the algorithm where the text is to mock a 100% accuracy (this is known as the ground truth) just in this component, and analyse how the global accuracy changes.

We simulate if our text detection system was 100% accurate, so we're feeding the character segmentation module with 100% accurate data now. How does this change the accuracy of the overall system?

![Ceiling Analysis: What Part of the Pipeline to Work on Next Example I](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Ceiling-Analysis%20What%20Part%20of%20the%20Pipeline%20to%20Work%20on%20Next%20Example%20I.png)

- Say, accuracy goes up to 89% (from 72%).
- Next, we do the same for the character segmentation
- And say, accuracy goes up to 90% now.
- Finally doe the same for character recognition.
- And it goes up to 100%.

Having done this we can qualitatively show what the upside to improving each module would be:

- Perfect text detection improves accuracy by 17%.
- Perfect character segmentation would improve it by 1%.
- Perfect character recognition would improve it by 10%.

We could conclude then, that improving the text detection module would result in the biggest accuracy gain. The character segmentation module is not really worth working on (at least at the moment), and finally the character recognition might be worth working on, depending if it would be easy or not to improve this module.

The "ceiling" is that each module has a ceiling by which making it perfect, it would improve the system overall.

One final example is face recognition. Note that this is not how it's done in practice, but let's imagine we have the following pipeline:

![Ceiling Analysis: What Part of the Pipeline to Work on Next Example II](https:/raw.githubusercontent.com/rmolinamir/machine-learning-notes/main/docs/3-machine-learning-systems/5-application-example-photo-ocr/images/Ceiling-Analysis%20What%20Part%20of%20the%20Pipeline%20to%20Work%20on%20Next%20Example%20II.png)

How would you do ceiling analysis for this?

- Overall system is 85%.
- Perfect background -> 85.1%.
  - Not a crucial step.
- Perfect face detection -> 91%.
  - Most important module to focus on.
- Perfect eyes ->95%.
- Perfect Nose -> 96%.
- Perfect Mouth -> 97%.
- Perfect logistic regression -> 100%.

Finally, a cautionary tale. Two engineers spent 18 months improving the background preprocessing component of this pipeline. It turned out that it had no impact on the overall performance, so they could have saved three years of man power if they'd done ceiling analysis first.
