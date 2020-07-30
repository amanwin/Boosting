# Boosting
Boosting is one of the most powerful ideas introduced in the field of machine learning in the past few years. It was first introduced in 1997 by Freund and Schapire in the popular algorithm, AdaBoost. It was originally designed for classification problems. Since its inception, many new boosting algorithms have been developed those tackle regression problems also and have become famous as they are used in the top solutions of many Kaggle competitions. We will go through the concepts of the most popular boosting algorithms - AdaBoost, Gradient Boosting and XGBoost in this module.

## Introduction to Boosting
In this session, we will go through the basics of Boosting and move on to the AdaBoost algorithm. Let's start with the motivation of studying Boosting and look at a brief overview of the technique.

We will discuss the following different boosting methods:
1. Adaptive Boosting
2. Gradient Boosting
3. XGBoost

The key idea of boosting is to create an ensemble which makes high errors only on the less frequent data points.

![title](img/expected-error.png)

Let's solve a question to understand the concept of expected error. <br/>
Suppose you are given two models who are expected to make a certain error on some data points. The probability of occurrence of the data points are provided in the following table:

![title](img/table.JPG)

The expected error of a model is the sum of errors weighted by the probabilities of the data points on which the errors occur. Our aim is to choose a model which will minimise the total expected error.

**Expected Error**
The expected errors of models 1 and 2 are respectively.

![title](img/table-er.JPG)

Boosting leverages the fact that we can build a series of models specifically targeted at the data points which have been incorrectly predicted by the other models in the ensemble. If a series of models keep reducing the average error, we will have an ensemble having extremely high accuracy.

![title](img/loss-function.JPG)

We learned that boosting is a way of generating a strong model from a weak learning algorithm. Note that here SVM, regression, and other techniques are algorithms which are used to create models. As mentioned about the function of the algorithm. It is to minimize the loss.

At this point, it is important to understand the loss functions for regression and classification problems are different. Until now, we have defined the error function for a regression setting as the sum of squared difference between the actual and the predicted values while the misclassification rate as the error function for a classification problem. We will see in the upcoming segments how the loss function is a modification of these error functions.

## Weak Learners
We shall learn the concept of weak learners and what role they play in the boosting learning algorithm.

So, here we see that a weak learning algorithm produces a model that does marginally better than a random guess. A random guess has a 50% chance of being right. Hence, any such model shall have, say 60-70% chance of being correct and the final objective is to create a strong model by making an ensemble of such weak models.

In case of decision tree, we want to have high interaction between attribute, but we have place a regularization on the depth of the tree to be 2. Then this will produce a model which will be merely better than random guess. But what we will do is we will somehow combine all these models to produce a model which is strong enough.
We saw an example of how to create a weak learner by applying a regularisation rule on the decision tree algorithm. 

Now that we understand what weak learners are, let’s see how they are combined to form an ensemble.

 