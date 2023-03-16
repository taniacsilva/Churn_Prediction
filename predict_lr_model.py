import pickle
import sklearn
import argparse

from flask import Flask
from flask import request
from flask import jsonify

model_file = "model_C=1.0.bin"



def load_model(model_file):
    """This function loads the model from .bin file

        Args:
            model_file (string) : contains the name of the file where the model and the DictVectorizer is saved

        Return:
            dv (sklearn.feature_extraction._dict_vectorizer.DictVectorizer) : method for one hot encoding imported from .bin file
            model (sklearn.linear_model._logistic.LogisticRegression) : Logistic regression model trained imported from .bin file
    """
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # using with it is ensured that the file is closed

    return dv, model

def customer_info_from_mkt():
    """This function
    """

    print("Getting the data from Marketing")
    customer = request.get_json()

    return customer

app = Flask('churn')
@app.route('/predict', methods=['POST'])

def predict (dv, model, customer):
    print("Using the model")
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        "churn_probability" : float(y_pred),
        "churn" : bool(churn)   
        }

    churn = y_pred >= 0.5

    return jsonify(result)


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
    """This is the main function of this Linear Model Regression Implementation model"""
    args = parse_arguments()
    
    print("Loading the model")
    dv, model = load_model(model_file)

    customer = customer_info_from_mkt()
    predict(dv, model, customer)
      

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)