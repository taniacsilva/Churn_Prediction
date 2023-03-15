import pickle
import sklearn
import argparse

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
    model_file = "model_C=1.0.bin"
    dv, model = load_model(model_file)
    print(model)

# Try one example

    customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
    }
    
    X = dv.transform([customer])
    
    model.predict_proba(X)[0, 1]

if __name__ == '__main__':
    main()