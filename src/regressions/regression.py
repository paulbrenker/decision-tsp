import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def regression_model_predictions(X_train, y_train, X_verify, model):
    """
        Returning predicted data for traing data input and a regression input function
    """
    # building the model
    model.fit(X_train, y_train)

    # making predictions
    predictions = model.predict(X_verify)
    
    return predictions


def create_prediction_acuracy_maps(predictions, y_verify, results, errors):
    """
        Create a map that contains accuracy for given limits. The limits
        are defined as float keys within the results map.
    """
    errors['mean_squared_error'].append(mean_squared_error(y_verify, predictions))
    errors['mean_absolute_error'].append(mean_absolute_error(y_verify, predictions))
    
    for key, value in results.items():
        if isinstance(key, str):
            continue
        above_limit = np.array(predictions < (y_verify * (1+key)))
        under_limit = np.array(predictions > (y_verify * (1-key)))

        within_limits = np.logical_and(above_limit, under_limit)

        correct_predictions_rel = np.count_nonzero(within_limits)/len(y_verify)
        value.append(correct_predictions_rel)

    
    return results, errors

def get_instances(
    len_dataset=2000,
    partition_of_data_set=1/5,
    number_of_sets=50,
    limits = [0.0025, 0.005, 0.01, 0.015, 0.02]):
    
    train_set_size = np.linspace(1, len_dataset*partition_of_data_set, num=number_of_sets) / len_dataset
    results = {}
    for limit in limits:
        results[limit] = []
    errors = errors = {
    'mean_squared_error': [],
    'mean_absolute_error': []
    }
    return train_set_size, results, errors


def train_models(X, y, model, train_set_size, results, errors, test_size=0.3):
    for t in train_set_size:
        #creating training and test data
        X_train, X_verify, y_train, y_verify = train_test_split(
            X, y, train_size=t, test_size=test_size, random_state=101)
    
        # creating a regression model
        predictions = regression_model_predictions(X_train, y_train, X_verify, model)
    
        #adding devs
        results, errors = create_prediction_acuracy_maps(predictions, y_verify, results, errors)

    return predictions, y_verify, results, errors