# Exercise 4 + 5 - Classification

The goal of this exercise was to use python to first train one model with tuned hyperparameters in order to classify all symbols in the emnist handwriting dataset, and second, to train multiple models and choose the best one using a train-validate-test split.

## Prepping the data

1. To start, I installed and imported the os, string, random, numpy, pandas, matplotlib, seaborn, emnist, iPython.display, sklearn (ensemble, metrics, linear model, preprocessing, model selection), xgboost, and tensorflow (keras models + layers).
2. Next, I brought in the helper functions written in Practice 2: Hands-on Handwritten.
    - I created a new version of the display_metrics function that would allow me to display multiclass confusion matrices.
3. The data was then loaded in and split into subsets (a to g, digits, and upper vs. lower) using the included code. 
4. Then I brought in a dictionary from Practice 2 that could store the results from each model I tested. I created three versions of the dictionary, one that could store the Classify Symbols multiclass data, one that could store the upper vs. lower binary data, and one that could store the even vs. odd binary data. 

## Classify Symbols

1. For the Classify Symbols part of the exercise, I decided to go with a random forest classifier, since they have the ability to classify more than two classes.
2. To make sure my code was working, I created a 'baby' version of the test and training sets, with the train having 5000 data points and the test having 2500 data points.
3. I started by bringing in the random forest code from Practice 2, and manipulating it to work with multiclass data.
    - I changed the average argument in the precision_score, recall_score, and f1_score functions to 'macro', which find the unweighted mean of the respective metric for all classes
    - I added a labels argument to the confusion_matrix function so it would work with multiple classes.
4. I created a loop to test multiple hyper parameters using 5-fold cross validation, and set accuracy as the evaluation metric.
    - These were: 'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'criterion': ['log_loss', 'gini', 'entropy']
5. I tested the code on the baby training and test sets.
6. Once the code was working, I ran the hyperparameter testing loop on the full a-g cross validation set.
7. The best parameters were: 'n_estimators': [150], 'max_depth': [None], 'criterion': ['entropy']
    - Highest parameter given for both!
8. The final accuracy was 96.4%. Pretty good I think!

## Upper vs. Lower

1. The steps for this problem were similar to the last problem, except that I added an outer loop to also test three different models along with hyperparameters.
2. The models chosen were logistic regression, random forest, and xgboost.
    - Logistic regression since this is a binary problem, and logistic regression is good for those.
    - Random forest, since it uses bagging (running multiple trees in parallel then averaging the result), to hopefully get a more accurate result.
    - XGBoost, which uses boosting to find the weaknesses in each subsequent tree to come out the other end with the best model.
3. I made sure to scale the data so it would converge for logistic regression. I used the standard scalar method included in sklearn to do so, which standardizes based on the average of all the data points.
4. I also created hyperparameter tuning grids for all three models.
    - Logistic regression: 'max_iter': [1000, 2000, 3000]
        - Even with scaling, 1000 iterations was the minimum number I could use to get the logistic regression model to converge.
    - Random forest: 'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'criterion': ['log_loss', 'gini', 'entropy']
    - XGBoost: 'n_estimators': [50, 100, 200, 500], 'max_depth': [1, 2, 3, 4, 5, 6], 'eta': [0.1, 0.3]
5. The best model was XGBoost, with the hyperparameters: 'n_estimators': [500], 'max_depth': [6], 'eta': [0.3]
    - Again, highest given parameters for all three!
6. The final accuracy for this model was 84.3%, which makes sense since some upper vs. lowercase letters may be difficult to tell apart, such as "x" and "X" or "c" and "C".

## Even vs. Odd

1. For this model, I used the same steps, models, and hyperparameters as the upper vs. lower problem.
2. Unfortunatey, the larger data set killed my Jupyter kernel, so I cut it in half using the "sample" method and ran it again just fine! (It took nearly 5 hours to run.)
3. The same XGBoost model with the same hyperparameters ('n_estimators': [500], 'max_depth': [6], 'eta': [0.3]) won for this problem too, but the accuracy was much higher, at 99.3%.
    - This makes sense, since the data set was much larger, and numbers are pretty easy to tell apart from eachother (02468 vs. 13579 look fairly different).