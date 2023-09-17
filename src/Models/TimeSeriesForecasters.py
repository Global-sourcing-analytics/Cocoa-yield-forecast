# -*- coding: utf-8 -*-
"""
TimeSeriesForecasters module

This module provides classes and utilities for the purpose of preparing data for various types of models
as well as for training, optimizing and evaluating them.

Classes:

    StatTimeSeriesForecaster: This class provides forecasting capabilities using auto-arima as well as
    functionalities for data preparation, model training, testing, predicting and evaluation.

    RegressionTimeSeriesForecaster: This class provides forecasting capabilities using regression-oriented
    machine learning methods as well as functionalities for data preparation, model training, testing, predicting
    and evaluation. Provides also method for optimization via random or grid search.

    DeepTimeSeriesForecaster: This class provides forecasting capabilities using deep learning models as well as
    functionalities for data preparation, model training, testing, predicting and evaluation. Provides also method
    for optimization via random or grid search.

Usage examples:

    # Evaluate RandomForest model
    model = RandomForestRegressor()
    randTreesTimeSeriesForecaster = TimeSeriesForecasters.RegressionTimeSeriesForecaster(model)

    forest_model, forest_preds, forest_scores = randTreesTimeSeriesForecaster.prepare_data(
        data, 
        categorical_col_name="ExampleID", 
        targets=["Example_target_1", Example_target_2"], 
        steps_ahead=6).evaluate_model(
            cv_splits=5, 
            model_name="data_example_RF", 
            save_destination="./Results")

    # Build and optimize deep learning model
    DNN_params = {
        "neurons_layer": list(range(5, 100, 1)),
        "activation_layer": ["relu"],
        "learning_rate": [x/100000 for x in range(1, 101, 1)],
        "optimizer": ["Adam"],
        "batch_size": list(range(5, 50, 1))
    } 

    init_params = {
        "neurons_layer": 16,
        "activation_layer": 'relu',
        "learning_rate": 0.0005,
        "optimizer": 'Adam',
        "batch_size": 20
    }

    def build_model_DNN(input_dim, output_num, params):
        model = Sequential()
        model.add(Flatten(input_shape=input_dim))
        model.add(Dense(params["neurons_layer"], params["activation_layer"]))
        model.add(Dense(params["neurons_layer"], params["activation_layer"]))
        model.add(Dense(params["neurons_layer"], params["activation_layer"]))
        model.add(Dense(params["neurons_layer"], params["activation_layer"]))
        model.add(Dense(output_num, 'linear'))

        model.compile(loss=MeanSquaredError(), optimizer=params["optimizer"], metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

        return model

    deepDNNTimeSeriesForecaster = TimeSeriesForecasters.DeepTimeSeriesForecaster(
        input_dim=(1, 24), 
        output_num=12, # output_num = steps_ahead * len(targets)
        build_model=build_model_DNN, 
        params=init_params)
    deepDNNTimeSeriesForecaster.prepare_data(
        data, 
        categorical_col_name="ExampleID", 
        targets=["Example_target_1", Example_target_2"], 
        steps_ahead=6)

    DNN_results = deepDNNTimeSeriesForecaster.optimize_model(
        build_model_DNN, 
        DNN_params, 
        input_dim=(1, 24), 
        output_num=12,
        cv_splits=5,
        search_method="random", # For random grid search, otherwise "grid"
        num_samples=60, " Only for search_method = "random"
        lag=0, # Only for LSTM based models
        verbose=0)
    
Author: Cezary Bujak
"""

# Basic imports
import os
import itertools
import traceback
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.base import clone
from tensorflow.keras.models import clone_model
from math import sqrt


# Validation
from sklearn.model_selection import TimeSeriesSplit

# Metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

# Auto-arima
import pmdarima as pm

# Neural networks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping

# Neural networks optimization
from keras.models import clone_model
from tensorflow.keras.optimizers import SGD # SGD (Stochastic Gradient Descent)
from tensorflow.keras.optimizers import RMSprop # RMSprop (Root Mean Square Propagation)
from tensorflow.keras.optimizers import Adam # Adam (Adaptive Moment Estimation)
from tensorflow.keras.optimizers import Adadelta # Adadelta stands for "Adaptive Delta" 
from tensorflow.keras.optimizers import Adagrad # Adagrad stands for "Adaptive Gradient"
from tensorflow.keras.optimizers import Adamax # Adamax is a variant of the Adam optimizer
# that uses the infinity norm (max norm) instead of the L2 norm for weight updates.
from tensorflow.keras.optimizers import Nadam # Nadam (Nesterov-accelerated Adaptive Moment Estimation)

def wape(y_true, y_pred):
    """Prevent from dividing by 0 like in MAPE

    Parameters
    ----------
    y_true : pandas.DataFrame or numpy.Array
        Real values.
    y_pred : pandas.DataFrame or numpy.Array
        Predicted values.

    Returns
    -------
    numpy.Array
        Array with wape results.
    """

    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

class StatTimeSeriesForecaster:
    """This class provides forecasting capabilities using auto-arima as well as
    functionalities for data preparation, model training, testing, predicting and evaluation.
    """

    def __init__(self, model=None):
        """Class instance initialization method. Can pass with no parameters.

        Parameters
        ----------
        model : Object, optional
            Model to be passed, by default None
        """

        self.model = model
    
    def prepare_data(self, data, categorical_col_name, targets):
        """Prepare the time series data for forecasting.

        This method preprocesses the input data for time series forecasting, including
        handling categorical features, selecting target variables, and performing any
        necessary data transformations.

        Parameters
        ----------
        data : pandas.DataFrame
            The input time series data.
        categorical_col_name : str
            The name of categorical column in the data.
        targets : list (str)
            The name(s) of the target variable(s) to be forecasted.

        Returns
        -------
        self
            The instance of the time series forecaster with prepared data.
        """

        self.categorical_col_name = categorical_col_name
        self.targets = targets

        # Need to copy, otherwise is working weirdely on the parameters passed directly
        data = data.copy()
        
        # TODO: Here hardcoded year and month needs to be parameterised
        data["Date"] = [datetime(int(row["year"]), int(row["month"]), 1) for index, row in data.iterrows()]
        data["Date"] = pd.to_datetime(data["Date"])
        data.index = data["Date"]
        data[categorical_col_name] = data[categorical_col_name].astype("category")

        self.data = data

        return self
    
    def get_data(self):
        """Get data being used by model.

        Returns
        -------
        pandas.DataFrame
            The data being used by model.
        """

        return self.data
    
    def split_category_data(self, n_splits):
        """Split the prepared categorical time series data into subsets.

        This method divides the prepared time series data into 'n_splits' subsets using 
        sklearn.model_selection.TimeSeriesSplit method. The splits are based on the categorical column(s) 
        which means that, for each category, time continuity is maintained.

        Parameters
        ----------
        n_splits : int
            The number of splits to create.

        Returns
        -------
        dict
            A dictionary containing the splitted data.
        """

        data_dict = {f"iter_{i}": {"train": pd.DataFrame(), "test": pd.DataFrame()} for i in range(1, n_splits+1)}

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # For each categorical unique value from categorical column
        # Split the data for that categorical value into train and test subset
        # Then append it to the proper structure in the dictionary
        for category in self.data[self.categorical_col_name].unique():
            for i, (train_index, test_index) in enumerate(tscv.split(self.data[self.data[self.categorical_col_name] == category].index.values)):
                train_data = self.data[self.data[self.categorical_col_name] == category].iloc[train_index]
                test_data = self.data[self.data[self.categorical_col_name] == category].iloc[test_index]

                # Adding test data to the train data with leaving 6 rows for testing. This is specific to 
                # testing the statistics based models
                train_data = train_data.append(test_data.iloc[:-6, :])
                test_data = test_data.tail(6)

                data_dict[f"iter_{i+1}"]["train"] = data_dict[f"iter_{i+1}"]["train"].append(train_data, ignore_index=True)
                data_dict[f"iter_{i+1}"]["test"] = data_dict[f"iter_{i+1}"]["test"].append(test_data, ignore_index=True)
        
        return data_dict
    
    def get_scores(self, tbl_preds):
        """Calculate forecast scores and return pivot tables and summary DataFrames.

        This method takes the forecasted predictions and calculates forecast accuracy
        scores. It returns three components:
    
        1. 'tbl_preds_pivot': A pivot table of the forecasted predictions (time x targets for each month).
        2. 'category_scores': A summary DataFrame containing accuracy scores for each unique categorical value.
        3. 'target_scores': A summary DataFrame containing accuracy scores for each target variable.

        Parameters
        ----------
        tbl_preds : pandas.DataFrame
            A DataFrame containing forecasted predictions.

        Returns
        -------
        pandas.DataFrame
            A pivot table of forecasted predictions.
        pandas.DataFrame
            A DataFrame with accuracy scores by category.
        pandas.DataFrame
            A DataFrame with accuracy scores by target variable.
        """

        # Categorical variable related scores
        category_scores = {"Iter_n": [], self.categorical_col_name: [], "Target": [], "rmse": [], "wape": [], "r2": [], "mae": []}
        for i in tbl_preds["Iter_n"].unique():
            for category in tbl_preds[self.categorical_col_name].unique():
                for target in self.targets:
                    category_scores["Iter_n"].append(i)
                    category_scores[self.categorical_col_name].append(category)
                    category_scores["Target"].append(target)
                    category_scores["rmse"].append(sqrt(mse(tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][target], 
                    tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][f"{target}_pred"])))
                    category_scores["wape"].append(wape(tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][target], 
                    tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][f"{target}_pred"]))
                    category_scores["r2"].append(r2_score(tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][target], 
                    tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][f"{target}_pred"]))
                    category_scores["mae"].append(mae(tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][target], 
                    tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)][f"{target}_pred"]))
        
        # Pivoted dataframe (from (time x targets) to (time x targets per month of prediction))
        tbl_preds_pivot = pd.DataFrame()
        target_values = self.targets + [f"{target}_pred" for target in self.targets]
        for i in tbl_preds["Iter_n"].unique():
            for category in tbl_preds[self.categorical_col_name].unique():
                temp = tbl_preds[(tbl_preds["Iter_n"] == i) & (tbl_preds[self.categorical_col_name] == category)]
                temp = temp.reset_index()
                temp = temp.pivot(index=[self.categorical_col_name, "Iter_n"], columns='Date', values=target_values)
                temp.columns = [f"{col[0]}_+{i}" for col, i in zip(temp.columns.values, list(range(1, 7)) * len(target_values))]
                temp = temp.reset_index()
                tbl_preds_pivot = tbl_preds_pivot.append(temp)
        
        # Targets per month of prediction related scores
        targets_col = [f"{target}_+{i}" for target in self.targets for i in list(range(1, 7))]
        preds_col = [f"{target}_+{i}" for target in [f"{target}_pred" for target in self.targets] for i in list(range(1, 7))]
        target_scores = {"Iter_n": [], "target": [], "rmse": [], "wape": [], "r2": [], "mae": []}
        for i in tbl_preds_pivot["Iter_n"].unique():
            for target, pred in zip(targets_col, preds_col):
                target_scores["Iter_n"].append(i)
                target_scores["target"].append(target)
                target_scores["rmse"].append(sqrt(mse(tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][target], 
                tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][pred])))
                target_scores["wape"].append(wape(tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][target], 
                tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][pred]))
                target_scores["r2"].append(r2_score(tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][target], 
                tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][pred]))
                target_scores["mae"].append(mae(tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][target], 
                tbl_preds_pivot[tbl_preds_pivot["Iter_n"] == i][pred]))
        
        return tbl_preds_pivot, pd.DataFrame(category_scores), pd.DataFrame(target_scores)

    def train_model(self, train, test, lag_period=6, is_seasonal=True):
        """Train a time series forecasting model and generate predictions.

        This method trains a time series forecasting model on the training data and uses
        it to generate predictions on the test data. It allows specifying a lag period
        and whether the time series is seasonal.

        Parameters
        ----------
        train : pandas.DataFrame
            The training data for model training.
        test : pandas.DataFrame
            The test data for generating predictions.
        lag_period : int, optional
            The lag period for time series. If 1 then auto-arima is not seasonal automatically, by default 6.
        is_seasonal : bool, optional
            True if the time series exhibits seasonality, False otherwise, by default True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the generated predictions.
        pandas.DataFrame
            The test data (for further comparison and scores).
        """

        # TODO: Handle the hardcoded "year" and "month" columns
        train["Date"] = [datetime(int(row["year"]), int(row["month"]), 1) for index, row in train.iterrows()]
        train["Date"] = pd.to_datetime(train["Date"])
        train.index = train["Date"]
        train.drop(columns=["Date", "year", "month"], inplace=True)

        test["Date"] = [datetime(int(row["year"]), int(row["month"]), 1) for index, row in test.iterrows()]
        test["Date"] = pd.to_datetime(test["Date"])
        test.index = test["Date"]
        test.drop(columns=["Date", "year", "month"], inplace=True)

        all_preds = pd.DataFrame()

        for target in self.targets:
            #Standard ARIMA Model
            model = pm.auto_arima(train[target], 
                                m=lag_period, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                                seasonal=is_seasonal)
            
            preds = model.predict(test[target].shape[0])
            all_preds[f"{target}_pred"] = preds

        return all_preds, test

    def evaluate_model(self, cv_splits=5, lag=6, seasonal=True):
        """Evaluate a time series forecasting model using forward chaining cross-validation.

        This method assesses the performance of the forecasting model through
        cross-validation. It returns a table of evaluation metrics for each fold.

        Parameters
        ----------
        cv_splits : int, optional
            The number of cross-validation splits, by default 5
        lag : int, optional
            The lag period for time series. If 1 then auto-arima is not seasonal automatically, by default 6
        seasonal : bool, optional
            True if the time series exhibits seasonality, False otherwise, by default True

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing predictions versus reals for each fold.
        """

        self.data_dict = self.split_category_data(cv_splits)

        tbl_preds = pd.DataFrame()
        # For each split, get train and test data
        # For each unique categorical value preprocess train and test if necessary
        # and train the model, get predictions, save them to prepared DataFrame
        for i in range(1, cv_splits+1):
            train, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["test"]
            for j, farmid in enumerate(train[self.categorical_col_name].unique()):
                print(f"Iteration {i} | Time series {j} out of {len(train[self.categorical_col_name].unique())}")
                temp_train, temp_test = train[train[self.categorical_col_name] == farmid], test[test[self.categorical_col_name] == farmid]
                temp_train.drop(columns=[self.categorical_col_name], inplace = True)
                temp_test.drop(columns=[self.categorical_col_name], inplace = True)

                preds, real = self.train_model(temp_train, temp_test, lag_period=lag, is_seasonal=seasonal)

                outcome = real
                outcome["Tiny+Small_pred"] = preds["Tiny+Small_pred"].values
                outcome["Large+Mature_pred"] = preds["Large+Mature_pred"].values
                outcome["FarmID"] = farmid
                outcome["Iter_n"] = i
                tbl_preds = tbl_preds.append(outcome)
        
        return tbl_preds


class RegressionTimeSeriesForecaster:
    """This class provides forecasting capabilities using regression-oriented
    machine learning methods as well as functionalities for data preparation, model training, testing, predicting
    and evaluation. Provides also method for optimization via random or grid search.
    """

    def __init__(self, model):
        """Class instance initialization method. Needs predefined model from scikit-learn module.

        Parameters
        ----------
        model : Object
            Predefined model from scikit-learn module
        """

        self.model = model
    
    def prepare_data(self, data, categorical_col_name, targets, steps_ahead):
        """Prepare the time series data for forecasting.

        This method preprocesses the input data for time series forecasting, including
        handling categorical features, selecting target variables, and performing any
        necessary data transformations.

        Parameters
        ----------
        data : pandas.DataFrame
            The input time series data.
        categorical_col_name : str
            The name of categorical column in the data.
        targets : list (str)
            The name(s) of the target variable(s) to be forecasted.
        steps_ahead : int
            The number of time steps ahead for forecasting horizon.
        
        Returns
        -------
        self
            The instance of the time series forecaster with prepared data.
        """
        
        self.categorical_col_name = categorical_col_name
        self.targets = []

        # Need to copy, otherwise weirdly is working on the parameters passed directly
        data = data.copy()
        # TODO: Handle the hardcoded "year" and "month" columns
        data["Date"] = [datetime(int(row["year"]), int(row["month"]), 1) for index, row in data.iterrows()]
        data["Date"] = pd.to_datetime(data["Date"])
        data.index = data["Date"]
        data.drop(columns=["Date"], inplace=True)
        data[categorical_col_name] = data[categorical_col_name].astype("category")
        
        # For each target make the feature indicating the i-step ahead value
        for target in targets:
            for i in range(1, steps_ahead+1):
                data[f"{target}_+{i}"] = data.groupby(categorical_col_name)[target].shift(-i)
                self.targets.append(f"{target}_+{i}")
        data = data.dropna()

        self.data = data

        return self
    
    def split_category_data(self, n_splits):
        """Split the prepared categorical time series data into subsets.

        This method divides the prepared time series data into 'n_splits' subsets using 
        sklearn.model_selection.TimeSeriesSplit method. The splits are based on the categorical column(s) 
        which means that, for each category, time continuity is maintained.

        Parameters
        ----------
        n_splits : int
            The number of splits to create.

        Returns
        -------
        dict
            A dictionary containing the splitted data.
        """

        data_dict = {f"iter_{i}": {"train": pd.DataFrame(), "test": pd.DataFrame()} for i in range(1, n_splits+1)}

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # For each categorical unique value from categorical column
        # Split the data for that categorical value into train and test subset
        # Then append it to the proper structure in the dictionary
        for category in self.data[self.categorical_col_name].unique():
            for i, (train_index, test_index) in enumerate(tscv.split(self.data[self.data[self.categorical_col_name] == category].index.values)):
                train_data = self.data[self.data[self.categorical_col_name] == category].iloc[train_index]
                test_data = self.data[self.data[self.categorical_col_name] == category].iloc[test_index]

                data_dict[f"iter_{i+1}"]["train"] = data_dict[f"iter_{i+1}"]["train"].append(train_data, ignore_index=True)
                data_dict[f"iter_{i+1}"]["test"] = data_dict[f"iter_{i+1}"]["test"].append(test_data, ignore_index=True)
        
        return data_dict

    def get_data(self):
        """Get data being used by model.

        Returns
        -------
        pandas.DataFrame
            The data being used by model.
        """

        return self.data

    def random_search(self, param_grid, num_samples):
        """Perform random hyperparameter search for model tuning.

        This method conducts random hyperparameter search by randomly sampling
        'num_samples' combinations of hyperparameters from 'param_grid'.

        Parameters
        ----------
        param_grid : dict
            A dictionary specifying hyperparameter ranges to explore.
        num_samples : int
            The number of random hyperparameter combinations to try.

        Returns
        -------
        list (dict)
            A list of dictionaries, each containing a random combination of hyperparameters.
        
        Example
        -------
        forecaster = RegressionTimeSeriesForecaster(model)
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.01, 0.1, 1.0],
            'param3': ['a', 'b', 'c']
        }
        random_combinations = forecaster.random_search(param_grid, num_samples=10)
        # 'random_combinations' now contains 10 random hyperparameter combinations.
        """

        random_combinations = []

        while len(random_combinations) < num_samples:
            random_combination = {}
            for param, values in param_grid.items():
                random_value = random.choice(values)
                random_combination[param] = random_value

            if random_combination not in random_combinations:
                random_combinations.append(random_combination)

        return random_combinations

    def grid_search(self, param_grid):
        """Perform grid hyperparameter search for model tuning.

        This method conducts grid hyperparameter search by iterating through all
        possible combinations of hyperparameters specified in 'param_grid'.

        Parameters
        ----------
        param_grid : dict
            A dictionary specifying hyperparameter values to explore.

        Returns
        -------
        list (dict)
            A list of dictionaries, each containing a unique combination of hyperparameters.
        
        Example
        -------
        forecaster = RegressionTimeSeriesForecaster(model)
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.01, 0.1, 1.0],
            'param3': ['a', 'b', 'c']
        }
        grid_combinations = forecaster.grid_search(param_grid)
        # 'grid_combinations' now contains all possible hyperparameter combinations.
        """

        # Generate all combinations of parameters
        grid_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        return grid_combinations

    def optimize_model(self, params, cv_splits = 5, search_method = "random", num_samples = 60):
        """Optimize model hyperparameters using a search method.
        TODO: Method is using only mae score for evaluation. Needs to be changed.
        In paper optimisation did not result in improved prediction performance, 
        to be improved in further work.
        TODO: Some values hardcoded, needs to be changed as well.

        This method optimizes the model's hyperparameters using the specified search method
        ('random' or 'grid'). It performs cross-validated hyperparameter tuning and returns
        the tuning results as well as the best hyperparameters found.

        Parameters
        ----------
        params : dict
            A dictionary specifying hyperparameter ranges or values to explore
        cv_splits : int, optional
            The number of cross-validation splits for evaluation, by default 5
        search_method : str, optional
            The hyperparameter search method ('random' or 'grid'), by default "random"
        num_samples : int, optional
            The number of random combinations to try if using 'random' search, by default 60

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing hyperparameter tuning results.
        dict
            The best hyperparameters found during tuning.
        """
        
        list_mae = []
        if search_method == "random":
            all_params = self.random_search(params, num_samples)
            # Use cross validation to evaluate all parameters
            self.data_dict = self.split_category_data(cv_splits)
            for i, params in enumerate(all_params):
                try:
                    model = clone(self.model)
                    model.set_params(**params)
                    print(f"Searching: {i+1}/{len(all_params)}")
                    print(model.get_params())
                    temp_mae = []
                    for i in range(1, cv_splits+1):
                        train, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["test"]
                
                        # One-hot encoding  
                        train_dummies = pd.get_dummies(train, columns=[self.categorical_col_name])
                        test_dummies = pd.get_dummies(test, columns=[self.categorical_col_name])

                        Xtr = train_dummies.drop(columns=list(train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
                        ytr = train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

                        Xtest = test_dummies.drop(columns=list(test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
                        ytest = test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

                        model.fit(Xtr, ytr)

                        preds = model.predict(Xtest)

                        scores = {"rmse": sqrt(mse(ytest, preds)), "wape": wape(ytest, preds), "r2": r2_score(ytest, preds), "mae": mae(ytest, preds)}

                        print(f"iter_{i}: mae: {scores['mae']}")
                        
                        temp_mae.append(scores["mae"])
                    list_mae.append(temp_mae)
                except Exception:
                    print(traceback.format_exc())
                    continue
        elif search_method == "grid":
            all_params = self.grid_search(params)
            # Use cross validation to evaluate all parameters
            self.data_dict = self.split_category_data(cv_splits)
            for i, params in enumerate(all_params):
                try:
                    model = clone(self.model)
                    model.set_params(**params)
                    print(f"Searching: {i+1}/{len(all_params)}")
                    print(model.get_params())
                    temp_mae = []
                    for i in range(1, cv_splits+1):
                        train, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["test"]
                
                        # One-hot encoding  
                        train_dummies = pd.get_dummies(train, columns=[self.categorical_col_name])
                        test_dummies = pd.get_dummies(test, columns=[self.categorical_col_name])

                        Xtr = train_dummies.drop(columns=list(train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
                        ytr = train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

                        Xtest = test_dummies.drop(columns=list(test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
                        ytest = test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

                        model.fit(Xtr, ytr)

                        preds = model.predict(Xtest)

                        scores = {"rmse": sqrt(mse(ytest, preds)), "wape": wape(ytest, preds), "r2": r2_score(ytest, preds), "mae": mae(ytest, preds)}

                        print(f"iter_{i}: mae: {scores['mae']}")
                        
                        temp_mae.append(scores["mae"])
                    list_mae.append(temp_mae)
                except Exception:
                    continue

        # Tuning results
        tuning_results = pd.DataFrame(all_params)
        tuning_results['mae_mean'] = [j[0] for j in [[np.mean(i), np.std(i)] for i in list_mae]]
        tuning_results['mae_std'] = [j[1] for j in [[np.mean(i), np.std(i)] for i in list_mae]]

        best_params = all_params[np.argmin([j[0] for j in [[np.mean(i), np.std(i)] for i in list_mae]])]
        return tuning_results, best_params

    def train_model(self, data):
        """Train the time series forecasting model.

        This method trains the time series forecasting model using the provided data.

        Parameters
        ----------
        data : pandas.DataFrame
            The training data for model training.
        """

        # One-hot encoding  
        data_dummies = pd.get_dummies(data, columns=[self.categorical_col_name])

        #TODO: Hardcoded - to be changed
        Xdata = data_dummies.drop(columns=list(data_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
        ydata = data_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

        self.model.fit(Xdata, ydata)
    
    def predict(self, data):
        """Generate predictions using the trained time series forecasting model.

        This method takes the trained forecasting model and generates predictions
        for the provided data.

        Parameters
        ----------
        data : pandas.DataFrame
            The data for which predictions should be generated.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the generated predictions.
        """

        data_dummies = pd.get_dummies(data, columns=[self.categorical_col_name])

        #TODO: Hardcoded - to be changed
        Xdata = data_dummies.drop(columns=list(data_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))

        preds = self.model.predict(Xdata)

        temp_outcome = data[["FarmID", "year", "month"]].copy()
        for target in self.targets:
            temp_outcome[f"{target}_pred"] = preds[:, self.targets.index(target)]
        tbl_preds = temp_outcome

        return tbl_preds

    def save_model(self):
        """Get model for further use.

        Returns
        -------
        Object
            Model used in the instance of the class
        """

        return self.model

    def get_scores(self, tbl_preds):
        """Calculate forecast scores and return summary dataframe per iteration, category, target.

        This method takes the forecasted predictions and calculates forecast scores per iteration,
        category and target.

        Parameters
        ----------
        tbl_preds : pandas.DataFrame
            A DataFrame containing forecasted predictions.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with accuracy scores per cross-validation iteration, unique categorical value and target.
        """
        
        tbl_scores = {"Iter_n": [], "FarmID": [], "Target": [], "rmse": [], "wape": [], "r2": [], "mae": []}
        for i in tbl_preds["Iter_n"].unique():
            for farmid in tbl_preds[self.categorical_col_name].unique():
                for target in self.targets:
                    farmid_indexes = tbl_preds[tbl_preds["Iter_n"] == i].index[tbl_preds[tbl_preds["Iter_n"] == i][self.categorical_col_name] == farmid].tolist()
                    real = tbl_preds[tbl_preds["Iter_n"] == i].iloc[farmid_indexes, :][target]
                    preds = tbl_preds[tbl_preds["Iter_n"] == i].iloc[farmid_indexes, :][f"{target}_pred"]
                    tbl_scores["Iter_n"].append(i)
                    tbl_scores["FarmID"].append(farmid)
                    tbl_scores["Target"].append(target)
                    tbl_scores["rmse"].append(sqrt(mse(real, preds)))
                    tbl_scores["wape"].append(wape(real, preds))
                    tbl_scores["r2"].append(r2_score(real, preds))
                    tbl_scores["mae"].append(mae(real, preds))

        return pd.DataFrame(tbl_scores)

    def evaluate_model(self, cv_splits=5, model_name="test", save_destination="./Results/test"):
        """Evaluate the time series forecasting model using cross-validation and save results.

        This method assesses the performance of the trained forecasting model through
        cross-validation and returns the trained model, predictions, and evaluation scores.
        It also saves the evaluation results to the specified destination.

        Parameters
        ----------
        cv_splits : int, optional
            The number of cross-validation splits for evaluation, by default 5
        model_name : str, optional
            The name of the trained model, by default "test"
        save_destination : str, optional
            The directory path to save evaluation results, by default "./Results/test"

        Returns
        -------
        Object
            The trained time series forecasting model
        pandas.DataFrame
            A DataFrame containing the generated predictions
        pandas.DataFrame
            A DataFrame containing evaluation metrics
        """

        self.data_dict = self.split_category_data(cv_splits)

        tbl_preds = pd.DataFrame()
        for i in range(1, cv_splits+1):
            print(f"Iteration {i}")
            train, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["test"]
            
            # One-hot encoding  
            train_dummies = pd.get_dummies(train, columns=[self.categorical_col_name])
            test_dummies = pd.get_dummies(test, columns=[self.categorical_col_name])

            Xtr = train_dummies.drop(columns=list(train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
            ytr = train_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

            Xtest = test_dummies.drop(columns=list(test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1).columns))
            ytest = test_dummies.filter(regex=('^(Tiny\+Small_\+\d|Large\+Mature_\+\d)'), axis=1)

            self.model.fit(Xtr, ytr)

            preds = self.model.predict(Xtest)

            temp_outcome = test[["FarmID", "year", "month"] + self.targets].copy()
            temp_outcome["Iter_n"] = i
            for target in self.targets:
                temp_outcome[f"{target}_pred"] = preds[:, self.targets.index(target)]
            tbl_preds = tbl_preds.append(temp_outcome)

        tbl_scores = self.get_scores(tbl_preds)

        # Take Iter_n column to the first position in tbl_preds
        col = tbl_preds.pop('Iter_n')
        tbl_preds.insert(0, col.name, col)

        # Take Iter_n column to the first position in tbl_scores
        col = tbl_scores.pop('Iter_n')
        tbl_scores.insert(0, col.name, col)

        # Mark the results with the model name if provided by user
        if model_name:
            tbl_preds["Model"] = model_name
            col = tbl_preds.pop('Model')
            tbl_preds.insert(0, col.name, col)

            tbl_scores["Model"] = model_name
            col = tbl_scores.pop('Model')
            tbl_scores.insert(0, col.name, col)

        # Save the results to the right destination provided by user
        # Remember: script is using the relative destination
        if save_destination:
            if not os.path.exists(os.getcwd() + save_destination):
                os.makedirs(os.getcwd() + save_destination)

            tbl_preds.to_csv(os.getcwd() + save_destination + "/preds.csv")
            tbl_scores.to_csv(os.getcwd() + save_destination + "/scores.csv")

        return self.model, tbl_preds, tbl_scores


class DeepTimeSeriesForecaster:
    """This class provides forecasting capabilities using deep learning models as well as
    functionalities for data preparation, model training, testing, predicting and evaluation. Provides also method
    for optimization via random or grid search.
    """

    def __init__(self, input_dim, output_num, build_model, params):
        """Initialize a DeepTimeSeriesForecaster object.

        This constructor initializes a DeepTimeSeriesForecaster instance with the provided
        parameters and settings for building a deep learning-based time series forecasting model.

        Note:
        - 'input_dim' should match the dimension of your input data.
        - 'build_model' should be a callable function that returns the deep learning model.

        Parameters
        ----------
        input_dim : tuple
            The dimension of the input data
        output_num : int
            The number of output units or predictions to generate
        build_model : callable
            A function that builds the deep learning model
        params : dict
            A dictionary of initial hyperparameters for configuring the model
        """

        # Specified seed for reproducibility
        seed = 7
        tf.random.set_seed(seed)
        self.params = params
        
        # A list which makes it possible to pass optimizer and learning_rate together
        optimizers_list = {"SGD": SGD(learning_rate=self.params["learning_rate"]), "RMSprop": RMSprop(learning_rate=self.params["learning_rate"]), 
                    "Adam": Adam(learning_rate=self.params["learning_rate"]), "Adadelta": Adadelta(learning_rate=self.params["learning_rate"]), 
                    "Adagrad": Adagrad(learning_rate=self.params["learning_rate"]), "Adamax": Adamax(learning_rate=self.params["learning_rate"]), 
                    "Nadam": Nadam(learning_rate=self.params["learning_rate"])}
        
        temp_params = self.params.copy()
        temp_params["optimizer"] = optimizers_list[temp_params["optimizer"]]
        model = build_model(input_dim, output_num, temp_params)
        
        self.model = model

        print(self.params)
        print(self.model.summary())
    
    def prepare_data(self, data, categorical_col_name, targets, steps_ahead):
        """Prepare the time series data for forecasting.

        This method preprocesses the input data for time series forecasting, including
        handling categorical features, selecting target variables, and performing any
        necessary data transformations.

        Parameters
        ----------
        data : pandas.DataFrame
            The input time series data.
        categorical_col_name : str
            The name of categorical column in the data.
        targets : list (str)
            The name(s) of the target variable(s) to be forecasted.
        steps_ahead : int
            The number of time steps ahead for forecasting horizon.

        Returns
        -------
        self
            The instance of the time series forecaster with prepared data.
        """

        self.categorical_col_name = categorical_col_name
        self.targets = []

        # Need to copy, otherwise weirdly is working on the parameters passed directly
        data = data.copy()
        data["Date"] = [datetime(int(row["year"]), int(row["month"]), 1) for index, row in data.iterrows()]
        data["Date"] = pd.to_datetime(data["Date"])
        data.index = data["Date"]
        data[categorical_col_name] = data[categorical_col_name].astype("category")

        day = 60*60*24
        year = 365.2425*day

        data["Seconds"] = data["Date"].map(pd.Timestamp.timestamp)

        # Creating date related fields standardized for the purpose of deep learning
        data['Year sin'] = np.sin(data['Seconds'] * (2 * np.pi / year))
        data['Year cos'] = np.cos(data['Seconds'] * (2 * np.pi / year))

        data.drop(columns=["Seconds", "Date"], inplace=True)

        # Data standarization - saving the mean and std for each column for further reversal processing
        columns_to_scale = data.drop(columns=[self.categorical_col_name, "year", "month", "Year sin", "Year cos"]).columns.values
        self.data_mean = data[columns_to_scale].mean()
        self.data_std = data[columns_to_scale].std()

        data[columns_to_scale] = (data[columns_to_scale] - self.data_mean)/self.data_std

        for target in targets:
            for i in range(1, steps_ahead+1):
                data[f"{target}_+{i}"] = data.groupby(categorical_col_name)[target].shift(-i)
                self.targets.append(f"{target}_+{i}")
        
        data = data.dropna()

        self.data = data

        return self
    
    def get_data(self):
        """Get data being used by model.

        Returns
        -------
        pandas.DataFrame
            The data being used by model.
        """

        return self.data

    def split_category_data(self, n_splits):
        """Split the prepared categorical time series data into subsets.

        This method divides the prepared time series data into 'n_splits' subsets using 
        sklearn.model_selection.TimeSeriesSplit method. The splits are based on the categorical column(s) 
        which means that, for each category, time continuity is maintained.

        Parameters
        ----------
        n_splits : int
            The number of splits to create.

        Returns
        -------
        dict
            A dictionary containing the splitted data.
        """

        data_dict = {f"iter_{i}": {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()} for i in range(1, n_splits+1)}

        # Tworzenie TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # For each categorical unique value from categorical column
        # Split the data for that categorical value into train, validation and test subset
        # Then append it to the proper structure in the dictionary
        for category in self.data[self.categorical_col_name].unique():
            for i, (train_index, test_index) in enumerate(tscv.split(self.data[self.data[self.categorical_col_name] == category].index.values)):
                train_data = self.data[self.data[self.categorical_col_name] == category].iloc[train_index]
                valid_data = self.data[self.data[self.categorical_col_name] == category].iloc[test_index]
                test_data = pd.DataFrame(valid_data.tail(6))
                valid_data = valid_data.iloc[:-6, :]

                data_dict[f"iter_{i+1}"]["train"] = data_dict[f"iter_{i+1}"]["train"].append(train_data, ignore_index=True)
                data_dict[f"iter_{i+1}"]["valid"] = data_dict[f"iter_{i+1}"]["valid"].append(valid_data, ignore_index=True)
                data_dict[f"iter_{i+1}"]["test"] = data_dict[f"iter_{i+1}"]["test"].append(test_data, ignore_index=True)
        
        return data_dict

    def df_to_X_y_lags(self, df, window_size=12):
        """Convert a time series DataFrame to input-output pairs with adjusted time window.

        This method transforms a time series DataFrame into input-output pairs for time series
        forecasting with lagged features. It returns NumPy arrays of input sequences (X) and
        corresponding output sequences (y) where X has shape (number of records, time window size, number of features) and
        y has shape (number of records, number of targets).

        Parameters
        ----------
        df : pandas.DataFrame
            The time series DataFrame to be splitted.
        window_size : int, optional
            The size of the lagged window for creating input-output pairs, by default 12

        Returns
        -------
        numpy.ndarray
            An array of input sequences with lagged window
        numpy.ndarray
            An array of corresponding output sequences.
        """

        # Get the indexes of the specific target columns
        column_indexes = [df.columns.get_loc(col) for col in self.targets]
        df_as_np = df.to_numpy()
        exploatory_variables = np.delete(df_as_np, column_indexes, axis=1)
        X = []
        y = []
        for i in range(len(df_as_np) - window_size):
            row = [r for r in exploatory_variables[i:i+window_size]]
            X.append(row)
            label = df_as_np[i + window_size][column_indexes]
            y.append(label)
        return np.array(X), np.array(y)

    def df_to_X_y(self, df):
        """Convert a time series DataFrame to input-output pairs.

        This method transforms a time series DataFrame into input-output pairs for time series
        forecasting without lagged features. It returns NumPy arrays of input sequences (X)
        and corresponding output sequences (y). Where X has shape (number of records, 1, number of features) and
        y has shape (number of records, number of targets).

        Parameters
        ----------
        df : pandas.DataFrame
            The time series DataFrame to be splitted

        Returns
        -------
        numpy.ndarray
            An array of input sequences with lagged window
        numpy.ndarray
            An array of corresponding output sequences.
        """

        # Get the indexes of the specific target columns
        column_indexes = [df.columns.get_loc(col) for col in self.targets]
        # date_index = df.columns.get_loc("Date")
        # dates = df.to_numpy()[:, date_index]

        df_as_np = df.to_numpy()
        exploatory_variables = np.delete(df_as_np, column_indexes, axis=1)
        X = exploatory_variables.reshape((len(exploatory_variables), 1, exploatory_variables.shape[1]))
        y = df_as_np[:, column_indexes]

        return X.astype(np.float32), y.astype(np.float32)

    def train_model(self, train, valid, test, batch_size, lag=0, verbose=1):
        """Train a deep learning-based time series forecasting model.

        This method trains the deep learning-based forecasting model on the provided training data,
        validates it on the validation data, and generates predictions for the test data.

        Parameters
        ----------
        train : pandas.DataFrame
            Training DataFrame.
        valid : pandas.DataFrame
            Validation DataFrame
        test : pandas.DataFrame
            Test DataFrame
        batch_size : int
            The batch size for training the model
        lag : int, optional
            The lag period for time series, if > 0 time window is applied, by default 0
        verbose : int, optional
            Verbosity level (0, 1, or 2) for training progress, by default 1

        Returns
        -------
        pandas.DataFrame
            DataFrame containing model predictions on the test data
        pandas.DataFrame
            DataFrame containing the true target values for the test data
        pandas.DataFrame
            The training history of the model
        """

        # One-hot encoding  
        train_dummies = pd.get_dummies(train, columns=[self.categorical_col_name])
        valid_dummies = pd.get_dummies(valid, columns=[self.categorical_col_name])
        test_dummies = pd.get_dummies(test, columns=[self.categorical_col_name])

        if lag > 0:
            X_train, y_train = self.df_to_X_y_lags(train_dummies.drop(columns = ["year", "month"]), window_size=lag)
            X_valid, y_valid = self.df_to_X_y_lags(valid_dummies.drop(columns = ["year", "month"]), window_size=lag)
            X_test, y_test = self.df_to_X_y_lags(test_dummies.drop(columns = ["year", "month"]), window_size=lag)
        else:
            X_train, y_train = self.df_to_X_y(train_dummies.drop(columns = ["year", "month"]))
            X_valid, y_valid = self.df_to_X_y(valid_dummies.drop(columns = ["year", "month"]))
            X_test, y_test = self.df_to_X_y(test_dummies.drop(columns = ["year", "month"]))

        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

        # Stop if val_loss is not improving for 5 epochs straight
        callback = EarlyStopping(monitor='val_loss', patience=5)
        model = clone_model(self.model)
        optimizers_list = {"SGD": SGD(learning_rate=self.params["learning_rate"]), "RMSprop": RMSprop(learning_rate=self.params["learning_rate"]), 
                    "Adam": Adam(learning_rate=self.params["learning_rate"]), "Adadelta": Adadelta(learning_rate=self.params["learning_rate"]), 
                    "Adagrad": Adagrad(learning_rate=self.params["learning_rate"]), "Adamax": Adamax(learning_rate=self.params["learning_rate"]), 
                    "Nadam": Nadam(learning_rate=self.params["learning_rate"])}
        temp_params = self.params.copy()
        temp_params["optimizer"] = optimizers_list[temp_params["optimizer"]]
        model.compile(loss=MeanSquaredError(), optimizer=temp_params["optimizer"], metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
        history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_valid, y_valid), epochs=1000, callbacks=[callback], verbose=verbose)

        preds = model.predict(X_test).reshape(y_test.shape)

        preds = pd.DataFrame(preds)
        y_test = pd.DataFrame(y_test)

        preds.rename(columns={i: f"{t}_pred" for i, t in enumerate(self.targets)}, inplace=True)
        y_test.rename(columns={i: t for i, t in enumerate(self.targets)}, inplace=True)

        for target in self.targets:
            preds[f"{target}_pred"] = preds[f"{target}_pred"] * self.data_std[target.split("_")[0]] + self.data_mean[target.split("_")[0]]
            y_test[target] = y_test[target] * self.data_std[target.split("_")[0]] + self.data_mean[target.split("_")[0]]
        
        preds[["FarmID", "year", "month"]] = test[["FarmID", "year", "month"]].iloc[lag:, :].values
        y_test[["FarmID", "year", "month"]] = test[["FarmID", "year", "month"]].iloc[lag:, :].values
        
        return preds, y_test, history

    def random_search(self, param_grid, num_samples):
        """Perform random hyperparameter search for model tuning.

        This method conducts random hyperparameter search by randomly sampling
        'num_samples' combinations of hyperparameters from 'param_grid'.

        Parameters
        ----------
        param_grid : dict
            A dictionary specifying hyperparameter ranges to explore.
        num_samples : int
            The number of random hyperparameter combinations to try.

        Returns
        -------
        list (dict)
            A list of dictionaries, each containing a random combination of hyperparameters.
        
        Example
        -------
        forecaster = RegressionTimeSeriesForecaster(model)
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.01, 0.1, 1.0],
            'param3': ['a', 'b', 'c']
        }
        random_combinations = forecaster.random_search(param_grid, num_samples=10)
        # 'random_combinations' now contains 10 random hyperparameter combinations.
        """

        random_combinations = []

        while len(random_combinations) < num_samples:
            random_combination = {}
            for param, values in param_grid.items():
                random_value = random.choice(values)
                random_combination[param] = random_value

            if random_combination not in random_combinations:
                random_combinations.append(random_combination)

        return random_combinations

    def grid_search(self, param_grid):
        """Perform grid hyperparameter search for model tuning.

        This method conducts grid hyperparameter search by iterating through all
        possible combinations of hyperparameters specified in 'param_grid'.

        Parameters
        ----------
        param_grid : dict
            A dictionary specifying hyperparameter values to explore.

        Returns
        -------
        list (dict)
            A list of dictionaries, each containing a unique combination of hyperparameters.
        
        Example
        -------
        forecaster = RegressionTimeSeriesForecaster(model)
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.01, 0.1, 1.0],
            'param3': ['a', 'b', 'c']
        }
        grid_combinations = forecaster.grid_search(param_grid)
        # 'grid_combinations' now contains all possible hyperparameter combinations.
        """

        # Generate all combinations of parameters
        grid_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        return grid_combinations

    def optimize_model(self, build_model, param_grid, input_dim = (1, 91), output_num = 12, cv_splits = 5, search_method = "random", num_samples = 60, lag=0, verbose=0):
        """Optimize a deep learning-based time series forecasting model.

        This method performs hyperparameter optimization for a deep learning-based forecasting model.
        It uses the specified search method ('random' or 'grid') to find the best hyperparameters
        and returns a table of evaluation scores.

        Parameters
        ----------
        build_model : callable
            A function that builds the deep learning model
        param_grid : dict
            A dictionary specifying hyperparameter ranges or values to explore
        input_dim : tuple, optional
            The input dimension tuple (samples, features), by default (1, 91)
            If lag > 0 then tuple should be of the shape (samples, lag, features)
        output_num : int, optional
            The number of targets to generate, by default 12
        cv_splits : int, optional
            The number of cross-validation splits for evaluation, by default 5
        search_method : str, optional
            The hyperparameter search method ('random' or 'grid'), by default "random"
        num_samples : int, optional
            The number of random combinations to try if using 'random' search, by default 60
        lag : int, optional
            The lag period for time series, if > 0 time window is applied, by default 0
        verbose : int, optional
            Verbosity level (0, 1, or 2) for training progress visualization, by default 0

        Returns
        -------
        pd.DataFrame
            A DataFrame containing evaluation scores for hyperparameter tuning
        """

        # fix random seed for reproducibility
        list_rmse = []
        list_wape = []
        list_mae = []
        list_r2 = []
        if search_method == "random":
            all_params = self.random_search(param_grid, num_samples)
        elif search_method == "grid":
            all_params = self.grid_search(param_grid)
        # Use cross validation to evaluate all parameters
        self.data_dict = self.split_category_data(cv_splits)
        for i, params in enumerate(all_params):
            optimizer = {"SGD": SGD(learning_rate=params["learning_rate"]), "RMSprop": RMSprop(learning_rate=params["learning_rate"]), 
                        "Adam": Adam(learning_rate=params["learning_rate"]), "Adadelta": Adadelta(learning_rate=params["learning_rate"]), 
                        "Adagrad": Adagrad(learning_rate=params["learning_rate"]), "Adamax": Adamax(learning_rate=params["learning_rate"]), 
                        "Nadam": Nadam(learning_rate=params["learning_rate"])}
            params["optimizer"] = optimizer[params["optimizer"]]
            model = build_model(input_dim, output_num, params)
            self.model = model
            print(f"Searching: {i+1}/{len(all_params)}")
            print(params)
            print(self.model.summary())
            temp_rmse = []
            temp_wape = []
            temp_mae = []
            temp_r2 = []
            score_temp = {"rmse": [], "wape": [], "mae": [], "r2": []}
            for i in range(1, cv_splits+1):
                real_flatten = []
                preds_flatten = []
                train, valid, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["valid"], self.data_dict[f"iter_{i}"]["test"]

                preds, real, history = self.train_model(train, valid, test, params["batch_size"], lag, verbose=verbose)

                # Here collect preds and reals into one nice table
                temp_outcome = pd.concat([preds, real.drop(columns=["FarmID", "year", "month"])], axis=1)
                temp_outcome["Iter_n"] = i

                for target in self.targets:
                    real_flatten.append(list(temp_outcome[temp_outcome["Iter_n"] == i][target]))
                    preds_flatten.append(list(temp_outcome[temp_outcome["Iter_n"] == i][f"{target}_pred"]))
                
                real_flatten = pd.DataFrame([item for row in real_flatten for item in row])[0]
                preds_flatten = pd.DataFrame([item for row in preds_flatten for item in row])[0]

                temp_rmse.append(sqrt(mse(real_flatten, preds_flatten)))
                temp_wape.append(wape(real_flatten, preds_flatten))
                temp_mae.append(mae(real_flatten, preds_flatten))
                temp_r2.append(r2_score(real_flatten, preds_flatten))

            list_rmse.append(temp_rmse)
            list_wape.append(temp_wape)
            list_mae.append(temp_mae)
            list_r2.append(temp_r2)

        # Tuning results
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse_mean'] = [j[0] for j in [[np.mean(i), np.std(i)] for i in list_rmse]]
        tuning_results['rmse_std'] = [j[1] for j in [[np.mean(i), np.std(i)] for i in list_rmse]]
        tuning_results['mae_mean'] = [j[0] for j in [[np.mean(i), np.std(i)] for i in list_mae]]
        tuning_results['mae_std'] = [j[1] for j in [[np.mean(i), np.std(i)] for i in list_mae]]
        tuning_results['wape_mean'] = [j[0] for j in [[np.mean(i), np.std(i)] for i in list_wape]]
        tuning_results['wape_std'] = [j[1] for j in [[np.mean(i), np.std(i)] for i in list_wape]]
        tuning_results['r2_mean'] = [j[0] for j in [[np.mean(i), np.std(i)] for i in list_r2]]
        tuning_results['r2_std'] = [j[1] for j in [[np.mean(i), np.std(i)] for i in list_r2]]

        return tuning_results

    def get_scores(self, tbl_preds):
        """Calculate forecast scores and return summary dataframe per iteration, category, target.

        This method takes the forecasted predictions and calculates forecast scores per iteration,
        category and target.

        Parameters
        ----------
        tbl_preds : pandas.DataFrame
            A DataFrame containing forecasted predictions.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with accuracy scores per cross-validation iteration, unique categorical value and target.
        """

        tbl_scores = {"Iter_n": [], "FarmID": [], "Target": [], "rmse": [], "wape": [], "r2": [], "mae": []}
        for i in tbl_preds["Iter_n"].unique():
            for farmid in tbl_preds[self.categorical_col_name].unique():
                for target in self.targets:
                    farmid_indexes = tbl_preds[tbl_preds["Iter_n"] == i].index[tbl_preds[tbl_preds["Iter_n"] == i][self.categorical_col_name] == farmid].tolist()
                    real = tbl_preds[tbl_preds["Iter_n"] == i].iloc[farmid_indexes, :][target]
                    preds = tbl_preds[tbl_preds["Iter_n"] == i].iloc[farmid_indexes, :][f"{target}_pred"]
                    tbl_scores["Iter_n"].append(i)
                    tbl_scores["FarmID"].append(farmid)
                    tbl_scores["Target"].append(target)
                    tbl_scores["rmse"].append(sqrt(mse(real, preds)))
                    tbl_scores["wape"].append(wape(real, preds))
                    tbl_scores["r2"].append(r2_score(real, preds))
                    tbl_scores["mae"].append(mae(real, preds))
        
        return tbl_scores
    
    def evaluate_model(self, cv_splits=5, batch_size=20, lag=0, model_name="test", save_destination="/Results/test"):
        """Evaluate a deep learning-based time series forecasting model using cross-validation and save results.

        This method assesses the performance of the trained forecasting model through
        cross-validation, generates predictions, and saves evaluation results, including
        predictions and history.

        Parameters
        ----------
        cv_splits : int, optional
            The number of cross-validation splits for evaluation, by default 5
        batch_size : int, optional
            The batch size for training, by default 20
        lag : int, optional
            The lag period for time series, if > 0 time window is applied, by default 0
        model_name : str, optional
            The name of the trained model, by default "test"
        save_destination : str, optional
            The directory path to save evaluation results, by default "/Results/test"

        Returns
        -------
        Object
            The trained time series deep forecasting model
        pandas.DataFrame
            A DataFrame containing the generated predictions
        pandas.DataFrame
            A DataFrame containing evaluation metrics
        pandas.DataFrame
            A DataFrame containing training history
        """

        self.data_dict = self.split_category_data(cv_splits)

        tbl_preds = pd.DataFrame()
        tbl_history = pd.DataFrame()
        for i in range(1, cv_splits+1):
            print(f"Iteration {i} out of {cv_splits}")
            train, valid, test = self.data_dict[f"iter_{i}"]["train"], self.data_dict[f"iter_{i}"]["valid"], self.data_dict[f"iter_{i}"]["test"]

            preds, real, history = self.train_model(train, valid, test, batch_size, lag)

            # Here collect preds and reals into one nice table
            temp_outcome = pd.concat([preds, real.drop(columns=["FarmID", "year", "month"])], axis=1)
            temp_outcome["Iter_n"] = i
            tbl_preds = tbl_preds.append(temp_outcome)

            temp_history = pd.DataFrame(history.history)
            temp_history["Iter_n"] = i
            tbl_history = tbl_history.append(temp_history)
        
        # Get scores for the provided preds and reals (can be seperated function)
        tbl_scores = self.get_scores(tbl_preds)
        tbl_scores = pd.DataFrame(tbl_scores)
        tbl_history = pd.DataFrame(tbl_history)

        # return self.model, tbl_preds, tbl_scores
        # Take Iter_n column to the first position in tbl_preds
        col = tbl_preds.pop('Iter_n')
        tbl_preds.insert(0, col.name, col)

        # Take Iter_n column to the first position in tbl_scores
        col = tbl_scores.pop('Iter_n')
        tbl_scores.insert(0, col.name, col)

        # Take Iter_n column to the first position in tbl_scores
        col = tbl_history.pop('Iter_n')
        tbl_history.insert(0, col.name, col)

        # Mark the results with the model name if provided by user
        if model_name:
            tbl_preds["Model"] = model_name
            col = tbl_preds.pop('Model')
            tbl_preds.insert(0, col.name, col)

            tbl_scores["Model"] = model_name
            col = tbl_scores.pop('Model')
            tbl_scores.insert(0, col.name, col)

            tbl_history["Model"] = model_name
            col = tbl_history.pop('Model')
            tbl_history.insert(0, col.name, col)

        # Save the results to the right destination provided by user
        # Remember: script is using the relative destination
        if save_destination:
            if not os.path.exists(os.getcwd() + save_destination):
                os.makedirs(os.getcwd() + save_destination)

            tbl_preds.to_csv(os.getcwd() + save_destination + "/preds.csv")
            tbl_scores.to_csv(os.getcwd() + save_destination + "/scores.csv")
            tbl_history.to_csv(os.getcwd() + save_destination + "/history.csv")
        
        return self.model, tbl_preds, tbl_scores, tbl_history