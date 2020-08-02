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

## AdaBoost Algorithm
We will learn how the AdaBoost algorithm improves the performance by boosting the probability of some points while suppressing the probability of others. AdaBoost stands for Adaptive Boosting, was developed by Schapire and Freund, who later on won the 2003 Godel Prize for their work.

Let us consider a binary classification problem, wherein learning algorithm will produce a model given a training set and distribution.

![title](img/adaboost.JPG)

In above formula D is the distribution, T is the training set, p is the probability of data points.
So we start with a uniform distribution, so with each iteration the points which are continuously not handled by the learning algorithm will be assigned a high probablity, so that in subsequent iterations those points are considered and finally we will produce a ensemble of such models which will in turn have high accuracy.

We see here that the with each new model, the distribution of the data changes. By distribution, we mean that the weight assigned to each data point changes for the calculation of objective function that needs to be minimized.

![title](img/objective-function.JPG)

In other words, objective function is the expected value of the loss which transforms to the above-mentioned objective function when the distribution is not uniform. 

**Question**

![title](img/question.JPG)

![title](img/answer.JPG)

Let's now look at the questions we need to answer to create an AdaBoost learning algorithm.
So, there are essentially two steps involved in the AdaBoost algorithm:
1. Modify the current distribution to create a new distribution to generate a new model.
2. Calculation of the weights given to each of the models to get the final ensemble.

![title](img/steps_adaboost.JPG)

![title](img/summary.JPG)

**Additional Reading:** <br/>
A Short Introduction to Boosting' by Freund and Schapire (http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf)

### AdaBoost Distribution and Parameter Calculation
We saw that the two basic steps involved in the AdaBoost algorithm are:
1. Modify the current distribution to create a new distribution to generate a new model
2. Calculation of the weights given to each of the models to get the final ensemble

We will explore the first step of the AdaBoost process, i.e. how the distribution changes after every iteration and the intuition behind it.

The probability assigned to different data points in the AdaBoost algorithm is as follows:

![title](img/adaboost1.JPG)

![title](img/adaboost-summary.JPG)

Now, let's dive into the second step of the AdaBoost process in which we look at how we assign weights to the different models we create at each step.

![title](img/alpha.JPG)

Since our algorithm is a weak learner, which is capable of producing a model which is marginally better than the random guess. So probablity of error by the model which this algorithm produces must be less than 1/2. So 1 - epsilon_t will be more than 1/2 and epsilon_t will be less than  1/2.

![title](img/epsilon.JPG)

![title](img/adaboost2.JPG)

![title](img/adaboost-summary1.JPG)

**Practical advice**: Before you apply the AdaBoost algorithm, you should remove the Outliers. Since AdaBoost tends to boosts up the probabilities of misclassified points and there is a high chance that outliers will be misclassified, it will keep increasing the probability associated with the outliers and make the progress difficult. Some of the ways to remove outliers are:
* Boxplots
* Cook's distance
* Z-score.

**Additional Reading**
The AdaBoost algorithm uses the exponential loss function, 

![title](img/loss-function1.JPG)

 to develop the expressions for the probability distribution and the weights assigned to the tree. You can go through the link mentioned for the derivation. (https://mbernste.github.io/files/notes/AdaBoost.pdf)

 ### AdaBoost Lab
 The objective of this segment is to learn how to implement the AdaBoost algorithm in python. In this exercise, you have to go through the notebook attached to the page and answer the questions that follow.

 Download and implement the code in the following notebook to get an understanding of the AdaBoostClassifier and answer the following questions. Refer to the documentation (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier) of AdaBoostClassifier if needed

 You can try changing the number of trees in 'estimators = list(range(1, 50, 3))' to 'estimators = list(range(1, 200, 3))' and see if the acuracy increases. 

 Note that we have used 'accuracy_score' as the evaluation metric here. We can use other evaluation metrics(https://scikit-learn.org/stable/modules/model_evaluation.html) also like 'roc_auc_curve'.

 [AdaBoost Lab](dataset/Adaboost_cancer_prediction.ipynb)

 ## Gradient Boosting

 ### Introduction
 Until now, you have learned the original boosting algorithm, AdaBoost. 
 
 In this session, we will learn another popular boosting algorithm - **Gradient Boosting** and a modification of Gradient Boosting called **XGBoost** which is widely used in the industry. We will also briefly discuss some of the more advanced algorithms recently introduced
 
 Towards the end, we will go through the implementation of the different algorithms on a Kaggle Dataset in Python.

 ### Understanding Gradient Boosting
 In this session, we will go through the algorithm of gradient boosting in a regression setting. Let's start with understanding the broad idea of gradient boosting.

 ![title](img/gradient_boosting.JPG)

 ![title](img/gradient_boosting1.JPG)

 ![title](img/gradient_boosting2.png)

 ![title](img/gradient_boosting3.JPG)

 Now that you have got a brief understanding of how gradient boosting works, let's study what the gradient in gradient boosting is.

 ### Gradient in Gradient Boosting
 Let's now look at following which explains how modelling on the residuals helps the model improve its predictive power, i.e. reduce the error.

 ![title](img/gradient_boosting4.JPG)

 We got an intuition of how the gradient boosting process helps in reducing the error with each iteration. To summarise, the gradient boosting algorithm has two steps at each iteration:
 1. Find the residual and fit a new model on the residual
 2. Add the new model to the older model and continue the next iteration

Let’s now see how this can be seen as a **gradient descent problem**. Let’s now see how this can be seen as a gradient descent problem.

![title](img/gradient_boosting5.JPG)

We see that the negative gradient of the square loss function gives the residue. So the strategy of closing in on the residues at each step can be seen as taking a **step towards the negative gradient of the loss function**. We can observe here that gradient descent helps in generalising the boosting process.

Please **note** that this explanation and the aforementioned process of closing in on the residues is only valid for square error loss function.

### Gradient Boosting Algorithm
Let's digress from the regression setting for a bit, and look at a more formal setting of Gradient Boosting and the nomenclature that is used industry-wide.

![title](img/gradient_boosting6.png)

![title](img/gradient_boosting7.JPG)

![title](img/gradient_boosting-min.JPG)

![title](img/gradient_boosting-min1.JPG)

![title](img/gradient_boosting-min2.JPG)

![title](img/gradient_boosting-min3.JPG)

### XGBoost
XGBoost was first developed by Taiqi Chen and became famous in solving the Higgs Boson problem. XG Boost is an abbreviation for **extreme gradient boosting**.

Let us first recall how regression tree works. Each leaf of a regression tree predicts a numerical value y for a data point x, i.e. when you push a data point x through the tree, it ends up in one of the leaves and takes the value corresponding to that leaf.

![title](img/xgboost.png)

XGBoost stands for Extreme Gradient Boosting which involves Gradient Boosting on shallow decision trees. Like before, we add a shallow decision tree to a previous model to create a new model.

We now need to figure out how to generate this incremental shallow decision tree. XGBoost algorithm is a modification of the Gradient Boosting tree algorithm. Hence, to have a better understanding of the XGBoost process, let's go through the following gradient tree boosting algorithm. This Gradient Boosting tree algorithm is the same as the Gradient Boosting algorithm we discussed in the pervious just that the model which we fit at each iteration is a decision tree. Hence, there is a change in steps 3 and 4. Let's go through what those are:

![title](img/xgboost.JPG)

![title](img/xgboost1.JPG)

### Understanding XGBoost
Now that you have understood the Gradient Boosting tree algorithm, let's move to XGBoost. The upcoming text discusses the maths behind the XGBoost. It is expected that you write the equations to have a better understanding.

Extreme Gradient Boosting (XGBoost) is similar to gradient boosting framework but more efficient and advanced implementation of Gradient Boosting algorithm. XGBoost, short for “Extreme Gradient Boosting”, was introduced by Chen in 2014 and since it’s introduction, it has become one of the most popular algorithms in Machine Learning realm. Due to its robust accuracy, it has been widely used in machine learning competitions as well.

Both XGBoost and GBM follow the principle of gradient boosted trees, but XGBoost uses a more regularised model formulation to control overfitting, which gives it better performance, which is why it’s also known as ‘regularised boosting‘ technique.

### The mathematics behind the XGBoost model:
In an ideal machine learning model, the objective function is a sum of Loss function “L” and regularization “Ω”. Loss function controls the predictive power of the algorithm and regularization controls its simplicity

Objective Function : Training Loss + Regularization

The above Gradient Boosting Tree algorithm has the objective function as only the Training Loss while the XGBoost objective function constitutes of loss function evaluated over, all predictions and sum of regularization term for all predictors ('T' trees).

![title](img/xgboost2.JPG)

![title](img/xgboost3.JPG)

![title](img/mse.JPG)

### Taylor series approximation of the loss
The above described objective function can be approximated using Taylor series expansion and hence can be solved

![title](img/tailor.JPG)

![title](img/tailor1.JPG)

![title](img/tailor2.JPG)

### XGBoost Regularisation
![title](img/xgboost-regularisation.JPG)

**Hyperparameters - Learning Rate, Number of Trees and Subsampling**

![title](img/xgboost-regularisation1.JPG)

**Subsampling** is training the model in each iteration on a fraction of data (similar to how random forests build each tree).  A typical value of subsampling is 0.5 while it ranges from 0 to 1. In random forests, subsampling is critical to ensure diversity among the trees, since otherwise, all the trees will start with the same training data and therefore look similar. This is not a big problem in boosting since each tree is any way built on the residual and gets a significantly different objective function than the previous one. 

Apart from the above-mentioned hyperparameters, there are other parameters of decision trees like the depth of the tree, the minimum number of samples required for split etc.

Gradient Boosting has a tendency to overfit as we keep increasing the number of trees. We can keep a check on overfitting by consistently checking accuracy on validation data. This issue is automatically addressed in XGBoost as stated earlier.

#### Why is XGBoost so good?
1. Parallel Computing: when you run xgboost, by default, it would use all the cores of your laptop/machine enabling its capacity to do parallel computation
2. Regularization: The biggest advantage of xgboost is that it uses regularization and controls the overfitting and simplicity of the model which gives it better performance.
3. Enabled Cross Validation: XGBoost is enabled with internal Cross Validation function
4. Missing Values: XGBoost is designed to handle missing values internally. The missing values are treated in such a manner that if there exists any trend in missing values, it is captured by the model.
5. Flexibility: XGBoost id not just limited to regression, classification, and ranking problems, it supports user-defined objective functions as well. Furthermore, it supports user-defined evaluation metrics as well.

### Kaggle Practice Exercise
This is a practice session in which we will solve the Kaggle competition TalkingData AdTracking Fraud Detection Challenge(https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). The training set, as well as the solution, is provided in here. Please implement the code and try tuning the different parameters to achieve a better result, but first, let's understand why it is beneficial to participate in Kaggle competitions.

[Python Notebook](dataset/Boosting+-+TalkingData+Click+Fraud+.ipynb)

[Training Data](dataset/train_sample.csv)

#### Additional References
Installation Guide: XGBoost (https://xgboost.readthedocs.io/en/latest/build.html)