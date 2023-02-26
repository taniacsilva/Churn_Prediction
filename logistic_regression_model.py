# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

    print(data.head().T)


if __name__ == '__main__':
    main()