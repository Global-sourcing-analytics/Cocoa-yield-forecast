# Artificial intelligence methods for cocoa yield forecasting

## Overview

This repository contains code and notebooks for the master's thesis titled "Artificial intelligence methods for cocoa yield forecasting", which focuses on testing various predictive models for time series forecasting of cocoa pods crop. The project aims to explore different approaches to forecasting and evaluate their performance on real-world time series dataset.

## Project Structure

The project is organized as follows:

- **Tests** - tests performed during the work
- **Tests/Data** - tests on loading the data and EDA of agronomic data in profile_report.html
- **Tests/Models** - tests on various predictive models
- **src**
- **src/Data** - data preparation and EDA of agronomic data
- **src/Data/Farms_timeseries** - timeseries for different plantations
- **src/Data/RFECV_results** - results of RFECV in various csv files
- **src/Models** - containing most important content where various models were tested
- **src/Models/Results** - results of all tested models (**IMPORTANT** It does not contain sensitive data so the predictions made)

Important scripts:
- **src/BC_solutions/Databases.py** - script handling connection with specific SQL Server
- **src/Data/data_loader.py** - script handling all imports
- **src/Data/pods_EDA.ipynb** - EDA of agronomic data
- **src/Data/feature_selection.ipynb** - feature engineering and feature selection via RFECV
- **src/Models/TimeSeriesForecasters.py** - this module provides classes and utilities for the purpose of preparing data for various types of models
as well as for training, optimizing and evaluating them
- **src/Models/classes_test.ipynb** - testing various models using the TimeSeriesForecasters.py module and collected data
- **src/Models/ARIMA_VS_CatBoost.ipynb** - comparison of auto-arima (predicting aggregated data) versus CatBoost (predicting on granularity of single plantations)

## Python version and dependencies

- Python 3.9
- Jupyter Notebook
- Libraries and dependencies specified in `venv39ML.yaml`
