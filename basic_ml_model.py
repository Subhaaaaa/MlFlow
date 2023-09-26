import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    URL='winequality-red.csv'

    try:
        # Read data as data frame
        df = pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e

def evaluate(y_true,y_pred):

    # mae = mean_absolute_error(y_true,y_pred)
    # mse = mean_squared_error(y_true,y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    # r2 = r2_score(y_true,y_pred)

    # For forest models
    acc_score = accuracy_score(y_true,y_pred)

    # return mae,mse,rmse,r2,acc_score
    return acc_score

def main(n_estimators, max_depth):
    df = get_data()
    train, test = train_test_split(df, test_size=0.2)

    # Splliting of data
    X_train = train.drop(['quality'], axis=1)
    X_test = test.drop(['quality'], axis=1)
    y_train = train[['quality']]
    y_test = test[['quality']]

    # Model training 

    # linear_regression = ElasticNet()
    # linear_regression.fit(X_train,y_train)
    # pred = linear_regression.predict(X_test)

    rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)

    # Model Evaluation
    acc_score = evaluate(y_test, pred)

    # print('MAE = {},MSE = {},RMSE = {},R2 Score = {}'.format(mae,mse,rmse,r2))
    print('Accuracy score  = {}'.format(acc_score))

if __name__ == '__main__':

    # python basic_ml_model.py -n 100 -m 15
    args = argparse.ArgumentParser()
    args.add_argument('--n_estimators','-n', default=50, type=int)
    args.add_argument('--max-depth', '-m', default=5, type=int)
    parse_args=args.parse_args()

    try:
        main(n_estimators= parse_args.n_estimators, max_depth= parse_args.max_depth)
    except Exception as e:
        raise e