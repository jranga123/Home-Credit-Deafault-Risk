# Home-Credit-Deafault-Risk
### Overview
The course project is based on the Kaggle Competition on Home Credit Default Risk (HCDR). This project's purpose is to anticipate if a client will repay a loan. Home Credit uses a number of alternative data—including telco and transactional information—to estimate their clients' repayment ability in order to ensure that people who struggle to secure loans owing to weak or non-existent credit histories have a pleasant loan experience.

## Abstract
The purpose of this project is to create a machine learning model/deep learning model that can predict consumer behavior during loan repayment.

In this phase our goal is to build a multi-layer neural network model in Pytorch and use Tensorboard to visualize real-time training results.This phase focused on building high performance Neural Networks and monitoring error generalization with early stopping technique and evaluating the model performance by monitoring through loss functions such as CXE and Hinge Loss.We did built 2 models, First model contains one linear layer with Relu function for probability prediction and the second model contain one linear layer, one hidden layer with Relu function and sigmoid function for probability prediction. Using Tensorboard we visualize the CXE loss for training data for each epoch.

Our results in this phase for multi layer neural network model the AUC scores are 0.588 for train data and 0.5172 for test data. For single layer model the AUC score is 0.7558 for test data. For our submission in kaggle we received a public score of 0.512 and private score of 0.510.


## Project Description

Home Credit is an international non-bank financial institution, which primarily focuses on lending people money regardless of their credit history. Home credit groups aim to provide positive borrowing experience to customers, who do not bank on traditional sources for pertaining loans. Hence, Home Credit Group published a dataset on Kaggle website with the objective of identifying and solving unfair loan rejection.

The purpose of this project is to create a machine learning model that can predict consumer behavior during loan repayment. Our task in this phase is to create a pipeline to build a baseline machine learning model using Logistic Regression algorithm. The resultant model will be evaluated with various performance metrics in order to build a better model. Companies can be able to rely on the output of this model to identify if loan is at risk to default. The new model built would help companies to avoid losses and make significant profits and will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

The results of our machine learning pipelines will be measured using the follwing metrics;
* Confusion Matrix
* Accuracy Score
* Precision
* Recall
* F1 score
* AUC (Area Under ROC Curve)

The pipeline results will be compared and ranked using the appropriate measurements and the most efficient pipeline will be submitted to the HCDR Kaggle Competition.

*Workflow*

For this project, we are following the proposed workflow as mentioned below.	

<img src="https://imgur.com/LEFwBmr.png" />

## Data Description
Overview The full dataset consists of 7 tables. There is 1 primary table and 6 secondary tables.

### Primary Tables
* application_train

        This Primary table includes the application information for each loan application at Home Credit in one row. 
        This row includes the target variable of whether or not the loan was repaid. We use this field as the basis to determine the feature importance. The target variable is binary in nature based since this is a classification problem.
        * ‘1' - client with payment difficulties: he/she had late payment more than N days on at least one of the first M installments of the loan in our sample
        * '0' - all other cases
        The number of columns are 122. The number of data entries are 307,511.
    
* application_test

        This table includes the application information for each loan application at Home Credit in one row. The features are the same as the train data but exclude the target variable
        The number of columns are 121. The number of data entries are 48,744.
        
### Secondary Tables
* Bureau

        This table contains data points for all client's previous credits provided by other financial institutions that were reported to the Credit Bureau. There is one row for each previous credit, meaning a many-to-one relationship with the primary table. We could join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 17.The number of data entries are 1,716,428.

* Bureau Balance

        This dataset has the monthly balance history of every previous credit reported to the Credit Bureau. There is one row for each monthly balance, meaning a many-to-one relationship with the Bureau table. We could join it with bureau table by using bureau's ID, SK_ID_BUREAU.
        The number of columns are 3. The number of data entries are 27,299,925

* Previous Application

        This table contains records for all previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample. , meaning a many-to-one relationship with the primary table. We could join it with primary table by using current application ID, SK_ID_CURR.
        There are four types of contracts:
        a. Consumer loan(POS – Credit limit given to buy consumer goods)
        b. Cash loan(Client is given cash)
        c. Revolving loan(Credit)
        d. XNA (Contract type without values)
        The number of columns are 37. The number of data entries are 1,670,214

* POS CASH Balance

        This table includes a monthly balance snapshot of a previous point of sale or cash loan that the customer has at Home Credit. There is one row for each monthly balance, meaning a many-to-one relationship with the Previous Application table. We would join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 8. The number of data entries are 10,001,358.

* Credit Card Balance

        This table includes a monthly balance snapshot of previous credit cards the customer has with Home Credit. There is one row for each previous monthly balance, meaning a many-to-one relationship with the Previous Application table.We could join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 23. The number of data entries are 3,840,312
        
* Installments Payments

        This table includes previous repayments made or not made by the customer on credits issued by Home Credit. There is one row for each payment or missed payment, meaning a many-to-one relationship with the Previous Application table. We would join it with Previous Application table by using previous application ID, SK_ID_PREV, then join it with primary table by using current application ID, SK_ID_CURR.
        The number of columns are 8 . The number of data entries are 13,605,401
        
## EDA(Exploratory Data Analysis)

Exploratory data analysis is important to this project because it helps to understand the data and it allows us to get closer to the certainty that the future results will be valid, accurately interpreted, and applicable to the proposed solution.

In the phase-1 of our project eda helped us to look at the summary statistics on each table and focussing on missing data, Outliers and aggregate functions such as mean, median etc and visual representation of features for better understanding of the data.

For identifying missing data we made use of categorical and numerical features. Specific features have been visualized based on their correlation values. The highly correlated features were used to plot the density to evaluate the distributions in comparison to the target variable. We used different plots such as countplot, heatmap, densityplot, catplot etc for visualizing our analysis.

*Key Observations:*
* In terms of correlation of features with target variable there are no highly correlated features which makes it interesting for feature engineering phase.


## Feature Engineering and transformers
Feature Engineering is important because it is directly reflected in the quality of the machine learning model, This is because in this phase new features are created, transformed, dropped based on aggregator functions such as max, min, mean, sum and count etc. 

* Including Custom domain knowledge based features
* Creating engineered aggregated features
* Experimental modelling of the data
* Validating Manual OHE
* Merging all datasets
* Drop Columns with Missing Values

Domain knowledge-based features, which assist increase a model's accuracy, are an important aspect of any feature engineering process. The first step was to determine which of these were applicable to each dataset. Credit card balance after payment based on due amount, application amount average, credit average, and other new custom features were among them. Available credit as a percentage of income, Annuity as a percentage of income, Annuity as a percentage of available credit are all examples of percentages.

The next stage was to find the numerical characteristics and aggregate them into mean, minimum, and maximum values. During the engineering phase, an effort was made to use label encoding for unique values greater than 5. However, to reduce the amount of code required to perform the same functionality, a design choice was taken to apply OHE at the pipeline level for specified highly correlated variables on the final merged dataset.

Extensive feature engineering was carried out by experimenting with several modeling techniques using main, secondary, and tertiary tables before settling on an efficient strategy that used the least amount of memory. For Tier 3 tables bureau balance, credit card installment, installment payments, and point of sale systems cash balance, the first attempt entailed developing engineered and aggregated features. This was then combined with Tier 2 tables, such as prev application balance with credit card installment, installment payments, and point of sale systems cash balance, as well as aggregated features, to create prev application balance. Along with the core dataset application train, a flattened view comprising all of the aforementioned tables was integrated. As a result, there were a lot of redundant features that took up a lot of memory.

Attempt 2 involved creating custom and aggregated features for tier 3 tables and merging with tier 2 tables based on the primary key provided, which was later “extended” to the tier1 tables based on the additional aggregated columns. This approach created less duplicates, was optimized and occupied less memory by using a garbage collector after each merge.

A train dataframe was created by merging the Tier3, Tier2, and Tier1 datasets. There were extra precautions made to verify that no columns had more than 50% of the data missing.
The characteristics were engineered and included in the model with modest divides to assist test the model, however the accuracy was low. However, for Random forest and XGBoost, employing these combined features in conjunction with acceptable splits throughout the training face resulted in improved accuracy and reduced the risk of overfitting.
Label encoding for unique categorical values in all categorical fields, not just a few, will be the focus of future research and trials.

## Pipelines

*Phase-1*

Logistic regression model is used as a baseline Model, since it's easy to implement yet provides great efficiency. Training a logistic regression model doesn't require high computation power. We also tuned the regularization, tolerance, and C hyper parameters for the Logistic regression model and compared the results with the baseline model. We used 5 fold cross fold validation with hyperparameters to tune the model and apply GridSearchCV function in Sklearn.

Below is the workflow for the model pipeline.

*Phase-2*

<img src="https://imgur.com/YEBX92l.png" />

In Phase 1, we used the Logistic regression model as the baseline model since it didn't take a lot of computing resources and was simple to execute. We also used customized logistic models with a balanced dataset to increase the predictiveness of our model. In phase 2, we did look at different classification models to see if we can improve our forecast. Our main focus is on boosting algorithms, which are believed to be extremely efficient and relatively fast. The modeling workflow for phase 2 is depicted in the diagram below. We used XGBoost, RandomForest, and SVM in our research.


Below is the reason for choosing the mentioned models.

* XGBoost is one of the quickest implementations of gradient boosted trees. XGBoost is designed to handle missing values internally. This is helpful because there are many, many hyperparameters to tune.

* Random Forest is a tree-based machine learning algorithm that combines the output of multiple decision trees for making decisions. For each tree only a random subsample of the available features is selected for building the tree. Random Forest uses decision trees, which are more prone to overfitting.

* SVM performs similar to logistic regression when linear separation and performs well with non-linear boundaries depending on the kernel used. SVM is susceptible to overfitting/training issues depending on the kernel. A more complex kernel would overfit the model.

Boosting algorithms can overfit if the number of trees is very large. We did two submission in Kaggle, one using Voting Classifier and the other one with best classifier i.e. XGBoost. A Voting Classifier is a machine learning model that trains on an ensemble of various models and predicts an output based on their highest probability of chosen class as the output. We have chosen soft voting instead of hard voting since the soft voting predicts based on the average of all models.

*Phase-3*

Below is the workflow for the multi layer neural network model pipeline.

<img src="https://imgur.com/CO8HURE.png" />

Below is the pipeline for the Multi layer neural network model.

* Multi layer neural network model has been choosen as it might provide results with high accuracy on the given data, If we are able to identify the accurate no of hidden layers it would help to improve the AUC score on the test data.

## Hyperparameters Used

Below are the hyperparameters we used for training different models:

```
# Arrange grid search parameters for each classifier
params_grid = {
        'Logistic Regression': {
            'penalty': ('l1', 'l2'),
            'tol': (0.0001, 0.00001), 
            'C': (10, 1, 0.1, 0.01),
        }
    ,
        'Support Vector' : {
            'kernel': ('rbf','poly'),     
            'degree': (4, 5),
            'C': ( 0.0001, 0.001),   #Low C - allow for misclassification
            'gamma':(0.01,0.1,1)  #Low gamma - high variance and low bias
        }
    ,
        'XGBoost':  {
            'max_depth': [3,5], # Lower helps with overfitting
            'n_estimators':[200,300],
            'learning_rate': [0.01,0.1],
            'colsample_bytree' : [0.2], 
        },                      #small numbers reduces accuracy but runs faster 

        'RandomForest':  {
            'max_depth': [5,10],
            'max_features': [15,20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5],
            'bootstrap': [True],
            'n_estimators':[100]},
    }
```

### Best Parameters for All models

**Logistic Regression**

<img src="https://imgur.com/2273JWG.png" />

**Random Forest**

<img src="https://imgur.com/MeC8eGr.png" />

**XGBoost Classifier**

<img src="https://imgur.com/C8oVzNv.png" />

## Experimental results

**Traditional Models**
Below is the resulting table for the results on the given dataset.

<img src="https://imgur.com/4zHiMl0.png" />

**Deep Learning**

**Single Layer Neural Network**
<img src="https://imgur.com/kVEVZek.png" width="500" height="200" />
**Multi Layer Neural Network**
<img src="https://imgur.com/tjqyOXQ.png" width="500" height="200" />


## Feature Importance

**Random Forest**

<img src="https://imgur.com/GiGLir6.png" />

**XGBoost Classifier**

<img src="https://imgur.com/yWYpi87.png" />

## Leakage Problem:
Proper measures has been taken in order to reduce the leakage problems by making use of Cross validation folds while training the model and by allocating specific amount of data to the validation set in training data due to which we feel we did take appropriate measures to handle the leakage problem during all the phases especially while splitting the data we are making sure to drop the target variable and then try to predict the model. Also we did make sure to perform OHE for categorical attributes and standard scaler for numerical attributes along with these we made use of imputing techniques such as most frequent one for categorical attributes and mean for numerical attributes, Therefore considering all these steps we do feel we handled the data leakage problem throughout the project.

## Discussion of Results
Based on the models discussed above, XGBoost stood out as the best predictive model using the top 183 features with 75.37% ROC score and followed by Logistic regression and the worst performance by Multi layer neural network with 59.34% AUC score.

    * Logistic Regression : This model was chosen as the baseline model trained with both balanced and imbalanced dataset with feature engineering. The training accuracy for this model 70.05% and test accuracy as 69.84%. A 75.18% ROC score resulted with best parameters for this model.
    
    * XGBoost : By far this model resulted in the best model. Both in terms of timing and accuracy for the selected features and balanced dataset. The accuracy of the training and test are 86.00% and test 78.65%. Test ROC under the curve is 75.37%.

    * Random Forest : On our last decision tree models, Random Forest produced training accuracy of 85.90% and test accuracy of 78.77%. Test ROC score came out as 73.40%.
    
    * Multi layer Neural Network: By far this is the model which has been underperforming when compared to traditional models as it is resulting in 51.72% AUC score and for single layer it is resulting in 74.8% AUC score, Multi Layer neural network is underperforming due to lack of selecting best features and identifying the accurate no of hidden layers could be the possible reasons.
    
## TensorBoard Results

**Single Layer Neural Network Model results:**

<img src="https://imgur.com/Qs229pM.png" />

We can clearly notice that from the tensor board results that the loss for the train data has been converging at 300(approx) epoch out of 500 epochs.

**Multi layer Neural Network Model results below:**

<img src="https://imgur.com/DQrCUhu.png" />

We can clearly notice that from the tensor board results that the gradual decrease in loss for the train data during intial epochs. Thereafter the loss has been converged.

## Problems faced

The problem encountered apart from the accuracy of the model include:

* SVM model was unable to run on the cpu as it is running forever and resulting in kernel crash.
* An unstable platform for running Machine Learning Models and collaboration.
* Long running models and system crash was the one of the biggest challenge we faced during training the model.
* Resampling techniques didn't produce good results with ensemble models.
* Identifying the accurate no of hidden layers for multi layer neural network model.


## Conclusion

Our implementation using ML models to predict if an applicant will be able to repay a loan was successful. Extending from the phase-1's simple baseline model, data modelling with feature aggregation, feature engineering, and using various data preprocessing pipeline both increased & reduced efficiency of models. Models used for prediction were Logistic Regression , ensemble model approaches using gradient boosting, Xgboost,  Random forest and SVM. In the current phase we did try to implement Multi layer neural network model using Pytorch.

Our best performing model was XGBoost with the best AUC score of 75.37%, The lowest performing model is Multi layer neural network model with 51.72 % , Our best score in Kaggle submission for XGBoost submission is 0.72922 private and 0.72657 for public and for voting classifier the score is 0.75709 private and 0.75885 for public, However we did believe that Multi layer neural network model would result in higher AUC score, However it has been underperforming compared to traditional models and the AUC score is 0.510 private and 0.512 for public.


## Kaggle Submission
Please provide a screenshot of your best kaggle submission for traditional & Multi layer neural network model.   

<img src="https://imgur.com/eDJ18Dg.png" />

<img src="https://imgur.com/CFVFByV.png" />
