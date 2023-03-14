# Churn Prediction

This project is inspired in a ML Zoomcamp.

In this project I created a model to identify customers that are likely to churn or stoping to use a service. The dataset used was obtained from [kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as:

$$g(x_i)=y_i$$

Where, $y_i$ is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of churning.

With this project, I aim to build a model with historical data from customers and assign a score of the likelihood of churning.

I have followed the steps described:

* 👀 Prepare data

**Main Conclusions** : This step included data obtention and some procedures of data preparation, namely look at the data, make columns names and values look uniform, check if all the columns read correctly and check if the churn variable needs any preparation.


* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

**Main Conclusions** : I have splitted the dataset using Scikit-Learn into train, validation and test.


* 🌲 Exploratory Data Analysis (EDA)

**Main Conclusions** : Included checking missing values, look at the target variable (churn) and look at numerical and categorical variables. I have also performed feature importance analysis (as part of Exploratory Data Analysis) to identify which features affect the target variable:

* *Churn Rate* - Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.

* *Risk Ratio* - Ratio that evidences how likely customers within this group are to churn compared to the overall population .Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms.

* *Mutual Information* - Categorical Variables - How much can be learned about churn if the value of another variable is known - Gives information about the relative importance of the variables

* *Correlation* - Numerical Variables - Measures the importance of numerical variables.
    - Positive Correlation vs. Negative Correlation: 
        - When r is positive, an increase in x will increase y.
        - When r is negative, an increase in x will decrease y.
        - When r is 0, a change in x does not affect y.
    - Depending on its size, the dependency between both variables could be low, moderate, or strong:
        - LOW when r is between [0, -0.2[ or [0, 0.2[
        - MEDIUM when r is between [-0.2, -0.5[ or [2, 0.5[
        - STRONG when r is between [-0.5, -1.0] or [0.5, 1.0]

        Where, r is correlation coefficient


* 0️⃣1️⃣ One-hot Encoding

**Main Conclusions** : I have used Scikit Learn - Dict Vectorizer - to encode categorical features. This method represents each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise.


* 👩‍💻 Use Logistic Regression to identify customers that are likely to churn

**Main Conclusions** : Logistic regression is similar to linear regression because both models take into account the bias term and weighted sum of features. The difference between these models is that the output of linear regression is a real number, while logistic regression outputs a value between zero and one, applying the sigmoid function to the linear regression formula and is used for binary classification.

$$g(x_i)=Sigmoid(w_0+w_1\times(x_1)+w_2\times(x_2)+...+w_n\times(x_n))$$ 

$$Sigmoid = \frac{1}{(1+e^-z)}$$ 

In this way, the sigmoid function allows transforming a score into a probability.

Then, I have trained the model using Scikit Learn and applied it to the validation dataset.


* ✔ Evaluating the model 

 **Main Conclusions** : To evaluate the model I have used the accuracy metric $\frac{TP + TN}{TP + TN + FP + FN}$


* 🔎 Model Interpretation

**Main Conclusions** : Interpret the coefficients obtained for the logistic regression model.


* 🎆 Using the model

**Main Conclusions**: After finding the best model, it was trained with training and validation partitions (x_full_train) and the final accuracy was calculated on the test partition.


* ✔ Evaluating the model (Further Analysis)

 **Main Conclusions** : 
 * Accuracy: Measures the fraction of correct predictions. Specifically, it is the number of correct predictions divided by the total number of predictions. I have evaluated the accuracy of my model accross different thresholds to understand which one is the best one. The best decision cutoff, associated with the hightest accuracy (80%), was indeed 0.5. If the threshold is defined in 1, the model is dummy and predicts that no clients will churn, the accuracy would be 73%. The accuracy between this dummy model and my model is not very considerable, so it can be concluded that accuracy can not tell how good the model is because the dataset is unbalanced, which means that there are more instances from one category than the other, also known as class imbalance.
 
 * Confusion table is a way of measuring different types of errors and correct decisions that binary classifiers can make. Considering this information, it is possible to evaluate the quality of the model by different strategies. When comes to a prediction of an LR model, each falls into one of four different categories:
    * Prediction is that the customer WILL churn. This is known as the Positive class
        * And Customer actually churned - Known as a **True Positive** (TP)
        * But Customer actually did not churn - Knwon as a **False Positive** (FP)
    * Prediction is that the customer WILL NOT churn - This is known as the Negative class
        * Customer did not churn - **True Negative** (TN)
        * Customer churned - **False Negative** (FN)
    
    The accuracy corresponds to the sum of TN and TP divided by the total of observations

* Precision and Recall:
    * Precision indicates the fraction of positive predictions that are correct. It takes into account only the positive class (TP and FP - second column of the confusion matrix), as is stated in the following formula:
    $$P = \frac{TP}{(TP+FP)}$$
    * Recall measures the fraction of correctly identified postive instances. It considers parts of the postive and negative classes (TP and FN - second row of confusion table). The formula of this metric is presented below:
    $$R = \frac{TP}{(TP+FN)}$$

    
 * ROC (Receiver Operating Characteristic) curves consider Recall and FPR under all the possible thresholds.  The ROC curves need comparison against a point of reference to evaluate its performance, so the corresponding curves of random and ideal models are required. It is possible to plot the ROC curves with FPR and Recall scores vs thresholds, or FPR vs Recall. If the threshold is 0 or 1, the TPR and Recall scores are the opposite of the threshold (1 and 0 respectively), but they have different meanings.
    * FPR is the fraction of false positives (FP) divided by the total number of negatives (FP and TN - the first row of confusion matrix), and I want to minimize it. The formula of FPR is the following:
        $$\frac{FP}{(FP + TN)}$$
    * In the other hand, TPR or Recall is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion table), and I want to maximize this metric. 

* AUC (Area Under the Curve) can be interpreted as the probability that a randomly selected positive example has a greater score than a randomly selected negative example. To quantify how far/close the model is from ideal model I computed the AUC. The AUC of a random model is 0.5, while for an ideal one is 1.

 * Cross Validation: evaluating the same model on different subsets of a dataset, getting the average prediction, and spread within predictions. This method is applied in the parameter tuning step, which is the process of selecting the best parameter. In this algorithm, the full training dataset is divided into k partitions, I have trained the model in k-1 partitions of this dataset and evaluate it on the remaining subset. Then, I end up evaluating the model in all the k folds, and I have calculated the average evaluation metric for all the folds. In general, if the dataset is large, hold-out validation dataset strategy should be used. In the other hand, if the dataset is small or if I want to know the standard deviation of the model across different folds, cross-validation approach can use used.




