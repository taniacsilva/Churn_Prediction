# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sklearn

from sklearn.model_selection import train_test_split

#Data Preparation
def data_preparation (file_name):
    """This function imports the data from the csv file and performs data preparation
        Args:
            file_name (str): The filename
        Return:
            data (pandas.DataFrame): The pandas Dataframe
    """
    # Load the data
    data = pd.read_csv(file_name)
    
    # Cleaning the names of the columns
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    
    # Cleaning the columns content
    string_col=list(data.dtypes[data.dtypes == 'object'].index)

    for col in string_col:
        data[col]=data[col].str.lower().str.replace(" ", "_")

    # Convert totalcharges variable to numeric
    data.totalcharges = pd.to_numeric(data.totalcharges, errors='coerce')

    # Fill the missing values of totalcharges column with 0
    data.totalcharges = data.totalcharges.fillna(0)

    # Replace 'yes/no' in the target variable by '1/0'
    data.churn = (data.churn == 'yes').astype(int)

    return data, string_col

def split_train_val_test(data):
    """ This functions splits the dataset between train, validation and test

        Args:
            data (pandas DataFrame): list that contains the explanatory variables and objective variable
            
        Returns:
            set_used (list): list that contains x_train, x_val, x_test, y_train, y_val, y_test
    """
    
    # Create x variable with the explanatory variables and y variable with the objective variable
    x = data.loc[:,data.columns!='churn']
    y = data['churn']

    # Split dataset into full train (train and validation) - 80% - and test set - 20%.
    x_full_train, x_test, y_full_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

    # Split the full train dataset into train - 75% - and validation - 35%.
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_full_train, y_full_train, test_size=0.25, random_state=1)

    #Reset the index
    set_used = [x_train, x_val, x_test, y_train, y_val, y_test, x_full_train, y_full_train]
    for i in set_used:
        i = i.reset_index(drop=True)

    return set_used


def exploratory_data_analysis(set_used):
    """ This function will make some exploratory data analysis

        Args:
            set_used (list): list that contains x_train, x_val, x_test, y_train, y_val, y_test

        Return:
    
    """
    # Counts the number of customers that churn and the ones that will keep the service
    print(set_used[7].value_counts())

    # Churn Rate
    print(set_used[7].value_counts(normalize=True))
    global_churn_rate = set_used[7].mean() # similar result to previous line

    # Creation of a list of numerical variables
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = ['gender', 'seniorcitizen', 'partner', 'dependents','phoneservice', 'multiplelines', 'internetservice','onlinesecurity','onlinebackup', 'deviceprotection', 'techsupport','streamingtv', 'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']
    
    # Determine the number of unique attributes by categorical variable
    unique = set_used[6][categorical].nunique()

    return global_churn_rate, numerical, categorical, unique


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            file_name: name of the command line field to insert on the runtime

        Return:
            args: Stores the extracted data from the parser run
    """
    parser = argparse.ArgumentParser(description="Process all the arguments for this model")
    parser.add_argument("file_name", help="The csv file name")
    args = parser.parse_args()

    return args


def main():
    """This is the main function of this Linear Model Regression Implementation model
    """
    args = parse_arguments()
    
    data, string_col = data_preparation(args.file_name)
    set_used = split_train_val_test(data)

    global_churn_rate, numerical, categorical, unique = exploratory_data_analysis(set_used)

    print(unique)

if __name__ == '__main__':
    main()