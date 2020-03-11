# Random Forest or Decision Forest

A decision tree is a simple, deterministic data structure for modelling decision rules for a specific classifciation problem. Random forest is just an improvement over the top of the decision tree algorithm. The core idea behind Random Forest is to generate multiple small decision trees from random subsets of the data (hence the name “Random Forest”).

Each of the decision tree gives a biased classifier (as it only considers a subset of the data). They each capture different trends in the data. This ensemble of trees is like a team of experts each with a little knowledge over the overall subject but thourough in their area of expertise.

Now, in case of classification the majority vote is considered to classify a class. In analogy with experts, it is like asking the same multiple choice question to each expert and taking the answer as the one that most no. of experts vote as correct. 

In case of Regression, we can use the avg. of all trees as our prediction.In addition to this, we can also weight some more decisive trees high relative to others by testing on the validation data.

The low correlation between models is the key. Just like how investments with low correlations (like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts, uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions.

The reason for this wonderful effect is that the trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction. So the prerequisites for random forest to perform well are:

    1. There needs to be some actual signal in our features so that models built using those features do better than random          guessing.
    2. The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.


#### So how does random forest ensure that models diversify each other ?

    1. Bagging (Bootstrap Aggregation): 
    Each individual tree randomly samples from the dataset with replacement, resulting in different trees. 

    Notice that with bagging we are not subsetting the training data into smaller chunks and training each tree on a     different chunk. Rather, if we have a sample of size N, we are still feeding each tree a training set of size N (unless specified otherwise). But instead of the original training data, we take a random sample of size N with replacement. For example, if our training data was [1, 2, 3, 4, 5, 6] then we might give one of our trees the following list [1, 2, 2, 3, 6, 6]. Notice that both lists are of length six and that “2” and “6” are both repeated in the randomly selected training data we give to our tree (because we sample with replacement).

    2. Feature Randomness: 
    In a normal decision tree, when it is time to split a node, we consider every possible feature and pick the one that produces the most separation between the observations in the left node vs. those in the right node. In contrast, each tree in a random forest can pick only from a random subset of features. This forces even more variation amongst the trees in the model and ultimately results in lower correlation across trees and more diversification.

![Feature Randomness](https://miro.medium.com/max/1240/1*EemYMyOADnT0lJWSXmTDdg.jpeg)

Let’s go through a visual example — in the picture above, the traditional decision tree (in blue) can select from all four features when deciding how to split the node. It decides to go with Feature 1 (black and underlined) as it splits the data into groups that are as separated as possible.

Now let’s take a look at our random forest. We will just examine two of the forest’s trees in this example. When we check out random forest Tree 1, we find that it it can only consider Features 2 and 3 (selected randomly) for its node splitting decision. We know from our traditional decision tree (in blue) that Feature 1 is the best feature for splitting, but Tree 1 cannot see Feature 1 so it is forced to go with Feature 2 (black and underlined). Tree 2, on the other hand, can only see Features 1 and 3 so it is able to pick Feature 1.

So in our random forest, we end up with trees that are not only trained on different sets of data (thanks to bagging) but also use different features to make decisions.

## Application Steps:

    1. Data Acquisition
    
    2. Data Preprocessing
        a. Missing Data
        b. Converting Features:
            i. Convert features to numeric
            ii. Convert wide ranges to same scale
            iii. Missing Values
        c. Creating Categories
        c. Creating new Features
        
    3. Check correlation
    
    4. Feature Importance
    
    5. Hyperparameter Tuning
    
    6. Model Evaluation
        a. Confusion Matrix
        b. Relevance: Precision and Recall
        c. F-Score
        d. Precision Recall Curve
        e. ROC AUC Curve
        f. ROC AUC Score


### Feature Importance: 
Another great quality of random forest is that they make it very easy to measure the relative importance of each feature. Sklearn measure a features importance by looking at how much the treee nodes, that use that feature, reduce impurity (entropy) on average (across all trees in the forest). It computes this score automatically for each feature after training and scales the results so that the sum of all importances is equal to 1. 

### Hyperparameter Tuning
The best way to think about hyperparameters is like the settings of an algorithm that can be adjusted to optimize performance, just as we might turn the knobs of an AM radio to get a clear signal

While model parameters are learned during training — such as the slope and intercept in a linear regression — hyperparameters must be set by the data scientist before training. In the case of a random forest, hyperparameters include the number of decision trees in the forest and the number of features considered by each tree when splitting a node.

Hyperparameter tuning relies more on experimental results than theory, and thus the best method to determine the optimal settings is to try many different combinations evaluate the performance of each model. However, evaluating each model only on the training set can lead to one of the most fundamental problems in machine learning: overfitting. The standard procedure for hyperparameter optimization accounts for overfitting through cross validation.

#### Cross Validation
When we approach a machine learning problem, we make sure to split our data into a training and a testing set. In K-Fold CV, we further split our training set into K number of subsets, called folds. We then iteratively fit the model K times, each time training the data on K-1 of the folds and evaluating on the Kth fold (called the validation data)

![5 Fold Cross Validation](https://miro.medium.com/max/2000/0*KH3dnbGNcmyV_ODL.png)

For hyperparameter tuning, we perform many iterations of the entire K-Fold CV process, each time using different model settings. We then compare all of the models, select the best one, train it on the full training set, and then evaluate on the testing set. This sounds like an awfully tedious process! Each time we want to assess a different set of hyperparameters, we have to split our training data into K fold and train and evaluate K times. If we have 10 sets of hyperparameters and are using 5-Fold CV, that represents 50 training loops. Fortunately, as with most problems in machine learning, someone has solved our problem and model tuning with K-Fold CV can be automatically implemented in Scikit-Learn.

For additional details on implementation of hyper tuning on RF: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

### Model Evaluation
#### OOB

#### Confusion/Error Matrix:
A table that describes the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).

![Confusion Matrix Ex.](https://github.comcast.com/storage/user/18009/files/010db000-60db-11ea-8fda-94ddae1265d9)

In this confusion matrix, of the 8 actual cats, the system predicted that three were dogs, and of the five dogs, it predicted that two were cats. All correct predictions are located in the diagonal of the table (highlighted in bold), so it is easy to visually inspect the table for prediction errors, as they will be represented by values outside the diagonal.


#### Relevance: Precision & Recall
![Precision & Recall](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

Precision: Fraction of relevant instances among the retrieved instances. Also known as positive predicted value or precision of prediction

Recall: Fraction of total amount of relevant instances that were actually retrieved.


#### F-Score

You can combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns much more weight to low values. As a result of that, the classifier will only get a high F-score, if both recall and precision are high.

Unfortunately the F-score is not perfect, because it favors classifiers that have a similar precision and recall. This is a problem, because you sometimes want a high precision and sometimes a high recall. The thing is that an increasing precision, sometimes results in an decreasing recall and vice versa (depending on the threshold). This is called the precision/recall tradeoff. 


#### ROC AUC Curve
Another way to evaluate and compare your binary classifier is provided by the ROC AUC Curve. This curve plots the true positive rate (also called recall) against the false positive rate (ratio of incorrectly classified negative instances), instead of plotting the precision versus the recall.


![ROC AUC Curve](https://github.comcast.com/storage/user/18009/files/07049080-60dd-11ea-9c38-0cfd76fe0a8a)

The red line in the middel represents a purely random classifier (e.g a coin flip) and therefore your classifier should be as far away from it as possible. Our Random Forest model seems to do a good job.

Of course we also have a tradeoff here, because the classifier produces more false positives, the higher the true positive rate is.

#### ROC AUC Score

The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.

A classifiers that is 100% correct, would have a ROC AUC Score of 1 and a completely random classiffier would have a score of 0.5.


## Conclusion: Overview
The random forest is a classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.

#### What do we need in order for our random forest to make accurate class predictions?
1. We need features that have at least some predictive power. After all, if we put garbage in then we will get garbage out.
2. The trees of the forest and more importantly their predictions need to be uncorrelated (or at least have low correlations with each other). While the algorithm itself via feature randomness tries to engineer these low correlations for us, the features we select and the hyper-parameters we choose will impact the ultimate correlations as well.

## Resources:
A practice end to end Random Forest Example:
https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

How to Random Forest:
https://www.kaggle.com/niklasdonges/end-to-end-project-with-python

Hyperparamter tuning a random forest:
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

Interpreting a random forest: 
https://towardsdatascience.com/random-forest-3a55c3aca46d

Understanding random forests: from theory to practice: 
https://arxiv.org/pdf/1407.7502.pdf



