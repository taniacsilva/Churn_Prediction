# Churn Prediction

This project is inspired in a ML Zoomcamp.

In this project I created a model to identify customers that are likely to churn or stoping to use a service. The dataset used was obtained from [kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as:

$$g(x_i)=y_i$$

Where, $y_i$ is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of churning.

With this project, I aim to build a model with historical data from customers and assign a score of the likelihood of churning.

I have followed the following steps:

* 👀 Prepare data

**Main Conclusions** : This step included data obtention and some procedures of data preparation, namely look at the data, make columns names and values look uniform, check if all the columns read correctly and check if the churn variable needs any preparation.


* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

**Main Conclusions** : For each partition, feature matrices (X) and y vectors of targets were obtained. I have calculated the size of partitions and records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.


* 🌲 Exploratory Data Analysis (EDA)

**Main Conclusions** : Included checking missing values, look at the target variable (churn) and look at numerical and categorical variables. I have also performed feature importance analysis (as part of Exploratory Data Analysis) to identify which features affect the target variable:

* *Churn Rate* - Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others.

* *Risk Ratio* - Ratio that evidence how likely customers within this group are to churn compared to the overall population.Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms.

* *Mutual Information* - Categorical Variables - How much can be learned about churn if the value of another is known - Gives information about the relative importance of the variables

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

**Main Conclusions** : Binary Classification. Logistic regression is similar to linear regression because both models take into account the bias term and weighted sum of features. The difference between these models is that the output of linear regression is a real number, while logistic regression outputs a value between zero and one, applying the sigmoid function to the linear regression formula.

$g(x_i)=Sigmoid(w_0+w_1\times(x_1)+w_2\times(x_2)+...+w_n\times(x_n))$ 

$$Sigmoid = \frac{1}{(1+e^-z)}$$ 

In this way, the sigmoid function allows transforming a score into a probability.

Then, I have trained the model using Scikit Learn and applied it to the validation dataset.


* ✔ Evaluating the model with Accuracy

 **Main Conclusions** : To evaluate the model I have used the accuracy metric $\frac{TP + TN}{TP + TN + FP + FN}$


* 🔎 Model Interpretation

**Main Conclusions** : Interpret the coefficients obtained for the logistic regression model.


* 🎆 Using the model

**Main Conclusions**: After finding the best model, it was trained with training and validation partitions (x_full_train) and the final accuracy was calculated on the test partition.


