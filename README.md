# Churn Prediction

This project is inspired in a ML Zoomcamp.

In this project I created a model to identify customers that are likely to churn or stoping to use a service. The dataset used was obtained from [kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

I have followed the following steps:

* 👀 Prepare data

Main Conclusions : This step included read the data with pandas, look at the data, make columns names and values look uniform, check if all the columns read correctly and check if the churn variable needs any preparation.

* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

Main Conclusions : For each partition, feature matrices (X) and y vectors of targets were obtained. I have calculated the size of partitions and records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.

* 🌲 Exploratory Data Analysis (EDA)

Main Conclusions : Included checking missing values, look at the target variable (churn) and look at numerical and categorical variables. I have also performed feature importance analysis (as part of Exploratory Data Analysis) to identify which features affect our target variable

- Churn Rate - How likely customers within this group are to churn compared to the overall population (Difference)
- Risk Ratio - How likely customers within this group are to churn compared to the overall population (Ratio)
- Mutual Information - Categorical Varaibles - How much can be learned about one variable if the value of another is known - Gives information about the relative importance of the variables
- Correlation - Numerical Variables - Measures the importance of numerical variables. Positive Correlation means that if a variable increases, the churn rate increases as well.

* 0️⃣1️⃣ One-hot Encoding

Main Conclusions : I have used Scikit Learn - Dictionary Vectorizer - to encode categorical features.

* 👩‍💻 Use Logistic Regression to identify customers that are likely to churn

 Main Conclusions : Binary Classification. Logistic Regression is very similar to linear regression except that adds a sigmoid before computing the output.

 Sigmoid = $1\frac{1}{(1+e^-z)}$ 

Then, I have trained the model using Scikit Learn and applied it to the validation dataset.

* ✔ Evaluating the model with Accuracy

 Main Conclusions : I have calculated the accuracy $\frac{TP + TN}{TP + TN + FP + FN}$

* 🔎 Model Interpretation

Main Conclusions : Interpret the coefficients obtained for the logistic regression model.

* 🎆 Using the model

Main Conclusions: After finding the best model, it was trained with training and validation partitions (x_full_train) and the final accuracy was calculated on the test partition.


