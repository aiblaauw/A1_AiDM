# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os

os.chdir('C:/Users/Gebruiker/documents/leiden/advances in data mining')
location = "./ml-1m/ratings.dat"
np.random.seed(42)

def load_data(location):
    ratings=[]
    f = open(location, 'r')
    for line in f:
        data = line.split('::')
        ratings.append([int(z) for z in data[:3]])
    f.close()
    ratings=np.array(ratings)
    return ratings

folds = 5

def cross_validate(ratings, folds):
    sequences = [x % folds for x in range(len(ratings))]
    np.random.shuffle(sequences)
    test_rmse = 0
    train_rmse = 0
    test_mae = 0
    train_mae = 0
    for fold in range(folds):
        test_select = [x == fold for x in sequences]
        train_select = [x != fold for x in sequences]
        test_set = ratings[test_select]
        train_set = ratings[train_select]
        fold_test_rmse, fold_test_mae, fold_train_rmse, fold_train_mae = run_regression(train_set, test_set)
        test_rmse += fold_test_rmse
        train_rmse += fold_train_rmse
        test_mae += fold_test_mae
        train_mae += fold_train_mae
    mean_test_rmse = test_rmse / folds
    mean_train_rmse = train_rmse / folds
    mean_test_mae = test_mae / folds
    mean_train_mae = train_mae / folds
    return mean_test_rmse, mean_train_rmse, mean_test_mae, mean_train_mae
        
    
def global_average(ratings):
    ratings = ratings[:,2]
    mean_rating = np.mean(ratings)
    predictions = np.full(len(ratings), mean_rating)
    return predictions

def get_item_avg(ratings):
    unique, counts = np.unique(ratings[:,1], return_counts = True)
    item_avgs = dict()
    for x in zip(unique, counts): 
        item_avg = np.mean(ratings[np.where(ratings[:,1]==x[0])][:,2])
        item_avgs[x[0]] = item_avg
    return item_avgs

def get_user_avg(ratings):
    unique, counts = np.unique(ratings[:,0], return_counts = True)
    user_avgs = dict()
    for x in zip(unique, counts): 
        user_avg = np.mean(ratings[np.where(ratings[:,0]==x[0])][:,2])
        user_avgs[x[0]] = user_avg
    return user_avgs

def average_item(ratings):
    item_avgs = get_item_avg(ratings)
    
    predictions = list()
    for item in ratings[:,1]:
        predictions.append(item_avgs[item])
    predictions = np.array(predictions)
    return predictions


def average_user(ratings):
    user_avgs = get_user_avg(ratings)
    
    predictions = list()
    for item in ratings[:,0]:
        predictions.append(user_avgs[item])
    predictions = np.array(predictions)
    return predictions

def train_regression(ratings):
    user_avgs = get_user_avg(ratings)
    item_avgs = get_item_avg(ratings)
    
    averages = list()
    for row in ratings:
        averages.append([user_avgs[row[0]], item_avgs[row[1]], 1])
    averages = np.array(averages)
    alpha, beta, gamma = np.linalg.lstsq(averages, ratings[:,2], rcond=None)[0]
    return alpha, beta, gamma

def predict_regression(ratings, alpha, beta, gamma):    
    user_avgs = get_user_avg(ratings)
    item_avgs = get_item_avg(ratings)
    predictions = list()
    for row in ratings:
        prediction = alpha * user_avgs[row[0]] + beta * item_avgs[row[1]] + gamma
        if prediction < 1:
            prediction = 1
        elif prediction > 5:
            prediction = 5
        predictions.append(prediction)
    return np.array(predictions)

def run(train_ratings, test_ratings):
    test_predictions = average_user(test_ratings)
    test_ratings = test_ratings[:, 2]
    test_rmse = np.sqrt(np.mean(((test_predictions - test_ratings) ** 2)))
    test_mae = np.mean(np.abs(test_predictions - test_ratings))
    train_predictions = average_user(train_ratings)
    train_ratings = train_ratings[:, 2]
    train_rmse = np.sqrt(np.mean(((train_predictions - train_ratings) ** 2)))
    train_mae = np.mean(np.abs(train_predictions - train_ratings))
    return test_rmse, test_mae, train_rmse, train_mae

def run_regression(train_set, test_set):
    alpha, beta, gamma = train_regression(train_set)
    predictions = predict_regression(test_set, alpha, beta, gamma)
    test_ratings = test_set[:, 2]
    test_rmse = np.sqrt(np.mean(((predictions - test_ratings) ** 2)))
    test_mae = np.mean(np.abs(predictions - test_ratings))
    return test_rmse, test_mae, 0, 0
    
#train_ratings = train_set[:,2]
cross_validate(load_data('./ml-1m/ratings.dat'), 5)