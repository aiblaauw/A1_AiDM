# -*- coding: utf-8 -*-
"""
Assignment 1: Recommender systems,
Differences for different approaches to recommender systems were evaluated.

Testing two model-based  collaborative filtering techniques;
- Linear combination of the item-averages and user-averages
- Matrix factorization with gradient descent and regularization 

Accuracy was evaluated by RMSE and MAE error metrics, 
combined with n-fold cross validation for testing flexibility of the models
"""

import numpy as np


'''
Used to set location of the data
'''
#os.chdir('C:/Users/Gebruiker/documents/leiden/advances in data mining')
location = "ratings.dat"
np.random.seed(42)


'''
Loads the data into a numpy array
'''
def load_data(location):
    ratings = np.genfromtxt(location, usecols=(0, 1, 2), delimiter="::", dtype="int")
    return ratings

'''
Number of folds in k-fold cross validation
'''
folds = 5


'''
Divides the data in to 5 folds and assigns one of these five to be test and the other four to be training
Computes errors and determines the average error over the five times it's run
use run_regression(train_set, test_set) for the approach using regression
use run(train_set, test_set) for the three approaches using an average
'''
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
        
'''
Computes the global average and makes predictions for the train and the test set
'''  
def global_average(train_ratings, test_ratings):
    test_ratings = test_ratings[:,2]
    train_ratings = train_ratings[:,2]
    mean_rating = np.mean(train_ratings)
    test_predictions = np.full(len(test_ratings), mean_rating)
    train_predictions = np.full(len(train_ratings), mean_rating)
    return test_predictions, train_predictions

'''
Gets the average for each item and puts it into a dictionary
'''
def get_item_avg(ratings):
    unique, counts = np.unique(ratings[:,1], return_counts = True)
    item_avgs = dict()
    for x in zip(unique, counts): 
        item_avg = np.mean(ratings[np.where(ratings[:,1]==x[0])][:,2])
        item_avgs[x[0]] = item_avg
    item_avgs['global'] = np.mean(ratings[:,2])
    return item_avgs

'''
Gets the average for each user and puts it into a dictionary
'''
def get_user_avg(ratings):
    unique, counts = np.unique(ratings[:,0], return_counts = True)
    user_avgs = dict()
    for x in zip(unique, counts): 
        user_avg = np.mean(ratings[np.where(ratings[:,0]==x[0])][:,2])
        user_avgs[x[0]] = user_avg
    user_avgs['global'] = np.mean(ratings[:,2])
    return user_avgs

'''
Predicts based on the average value for each item
'''
def average_item(train_ratings, test_ratings):
    item_avgs = get_item_avg(train_ratings)
    
    test_predictions = list()
    for item in test_ratings[:,1]:
        if item in item_avgs:
            test_predictions.append(item_avgs[item])
        else:
            test_predictions.append(item_avgs['global'])
    test_predictions = np.array(test_predictions)
    
    train_predictions = list()
    for item in train_ratings[:,1]:
        train_predictions.append(item_avgs[item])
        if item not in item_avgs:
            train_predictions.append(item_avgs['global'])
    train_predictions = np.array(train_predictions)
    return test_predictions, train_predictions

'''
Predicts based on the average value for each user
'''
def average_user(train_ratings, test_ratings):
    user_avgs = get_user_avg(train_ratings)
    
    test_predictions = list()
    for user in test_ratings[:,1]:
        if user in user_avgs:
            test_predictions.append(user_avgs[user])
        else:
            test_predictions.append(user_avgs['global'])
    test_predictions = np.array(test_predictions)
    
    train_predictions = list()
    for user in train_ratings[:,1]:
        train_predictions.append(user_avgs[user])
        if user not in user_avgs:
            train_predictions.append(user_avgs['global'])
    train_predictions = np.array(train_predictions)
    return test_predictions, train_predictions

'''
Trains the regression coefficients alpha, beta and gamma, takes only the training set as input
'''
def train_regression(ratings):
    user_avgs = get_user_avg(ratings)
    item_avgs = get_item_avg(ratings)
    
    averages = list()
    for row in ratings:
        averages.append([user_avgs[row[0]], item_avgs[row[1]], 1])
    averages = np.array(averages)
    alpha, beta, gamma = np.linalg.lstsq(averages, ratings[:,2], rcond=None)[0]
    return alpha, beta, gamma

'''
Predicts ratings based on the regression coefficients and averages
'''
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

'''
Used to run one fold of the approaches using averages and returns the errors for the fold
global_average(train_ratings, test_ratings) for global average
average_item(train_ratings, test_ratings) for item averages
average_user(train_ratings, test_ratings) for user averages
'''  
def run(train_ratings, test_ratings):
    test_predictions, train_predictions = average_user(train_ratings, test_ratings)
    test_ratings = test_ratings[:, 2]
    test_rmse = np.sqrt(np.mean(((test_predictions - test_ratings) ** 2)))
    test_mae = np.mean(np.abs(test_predictions - test_ratings))
    train_ratings = train_ratings[:, 2]
    train_rmse = np.sqrt(np.mean(((train_predictions - train_ratings) ** 2)))
    train_mae = np.mean(np.abs(train_predictions - train_ratings))
    return test_rmse, test_mae, train_rmse, train_mae

'''
Used to run one fold of the regression approach and returns the errors for the fold
'''
def run_regression(train_set, test_set):
    alpha, beta, gamma = train_regression(train_set)
    test_predictions = predict_regression(test_set, alpha, beta, gamma)
    test_ratings = test_set[:, 2]
    test_rmse = np.sqrt(np.mean(((test_predictions - test_ratings) ** 2)))
    test_mae = np.mean(np.abs(test_predictions - test_ratings))
    train_predictions = predict_regression(train_set, alpha, beta, gamma)
    train_ratings = train_set[:, 2]
    train_rmse = np.sqrt(np.mean(((train_predictions - train_ratings) ** 2)))
    train_mae = np.mean(np.abs(train_predictions - train_ratings))
    return test_rmse, test_mae, train_rmse, train_mae

'''
Creates IxJ matrix which contains all the ratings and movies, also those not rated by some user 
'''
def reform_matrix(ratings):
    unique_users = np.unique(ratings[:,0])
    unique_items = np.unique(ratings[:,1])
    user_dict = dict()
    user_index = 0
    for user in unique_users:
        user_dict[user] = user_index
        user_index += 1
    
    item_dict = dict()
    item_index = 0
    for item in unique_items:
        item_dict[item] = item_index
        item_index += 1
    sparse_matrix = np.empty([len(user_dict), len(item_dict)])
    for row in ratings:
        sparse_matrix[user_dict[row[0]], item_dict[row[1]]] = row[2]
    
    return sparse_matrix
    
'''
Initialize matrices U and M randomly from a uniform distribution
'''
def initialize_matrices(num_users, num_items, num_features):
    U = np.random.rand(num_users, num_features)
    M = np.random.rand(num_features, num_items)
    return U, M

'''
Compute the errors, gradients and update the rows and columns of the matrix with estimates and feature matrices U and M
U = Feature matrix for Users
M = Feature matrix for Items
X = Sparse matrix containing all ratings
X_est = Matrix containing all the estimated ratings
lrate = Learning rate
regcof = Reguralization coefficient
test_values = values present in the test set
test_set = coordinates of items belonging to the test set
train_values = values present in the training set
train_set = coordinates of items belonging to the training set
'''
def update(U, M, X, X_est, lrate, regcof, test_values, test_set, train_values, train_set):
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] > 0:
                eij = X[i][j] - np.dot(U[i,:], M[:,j])
                U[i,:] = U[i,:] + (lrate * (2 * eij * M[:,j] - regcof * U[i,:]))
                M[:,j] = M[:,j] + (lrate * (2 * eij * U[i,:] - regcof * M[:,j]))
    X_est = np.matmul(U, M)
    '''
    #X_est = (((X_est - np.amin(X_est)) * 4) / (np.amax(X_est) - np.amin(X_est))) + 1
    SE = 0
    calculations = 0
    test_SE = 0
    test_calculations = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] > 0 and [i, j] not in test_set.T.tolist():
                SE += (X[i][j] - X_est[i][j]) ** 2
                calculations += 1
            elif X[i][j] > 0 and [i, j] in test_set.T.tolist():
                test_SE += (X[i][j] - X_est[i][j]) ** 2
                test_calculations += 1
    MSE = SE / calculations
    RMSE = np.sqrt(MSE)
    test_MSE = test_SE / test_calculations
    test_RMSE = np.sqrt(test_MSE)
    '''
    test_predictions = np.array([X_est[i][j] for i, j in test_set.T])
    train_predictions = np.array([X_est[i][j] for i, j in train_set.T])
    RMSE = np.sqrt(np.mean(((test_predictions - test_values) ** 2)))
    MAE = np.mean(np.abs(test_predictions - test_values))
    train_RMSE = np.sqrt(np.mean(((train_predictions - train_values) ** 2)))
    train_MAE = np.mean(np.abs(train_predictions - train_values))
    return train_RMSE, train_MAE, RMSE, MAE, U, M, X_est

'''
Runs the matrix factorization for a set number of iterations
'''
def run_matrix_fac(num_users, num_items, num_features, X, num_iterations, lrate, regcof, test_values, test_set, train_values, train_set):
    U, M = initialize_matrices(num_users, num_items, num_features)
    X_est = np.matmul(U, M)
    for i in range(num_iterations):
        train_RMSE, train_MAE, RMSE, MAE, U, M, X_est = update(U, M, X, X_est, lrate, regcof, test_values, test_set, train_values, train_set)
    return train_RMSE, train_MAE, RMSE, MAE

'''
Cross validation for matrix factorization
'''  
def cross_validate_matrix(X):
    ratings = np.array(X.nonzero())
    sequences = [x % folds for x in range(len(ratings[0]))]
    np.random.shuffle(sequences)
    sum_t_RMSE = 0
    sum_t_MAE = 0
    for fold in range(folds):
            test_select = [x == fold for x in sequences]
            test_set = ratings[:, test_select]
            test_values = np.array([X[i][j] for i,j in test_set.T])
            train_select = [x != fold for x in sequences]
            train_set = ratings[:, train_select]
            train_values = np.array([X[i][j] for i,j in train_set.T])
            for i,j in test_set.T:
                X[i][j] = 0
            fold_t_RMSE, fold_t_MAE, fold_RMSE, fold_MAE = run_matrix_fac(len(X), len(X[0]), 10, X, 75, 0.005, 0.05, test_values, test_set, train_values, train_set)
            print('RMSE of this fold is %5f and MAE this fold is %5f' % (fold_RMSE, fold_MAE))
            sum_t_RMSE += fold_t_RMSE
            sum_t_MAE += fold_t_MAE
    train_RMSE = sum_t_RMSE / 5
    train_MAE = sum_t_MAE / 5        
    print('Training RMSE is %5f, training MAE is %5f' % (train_RMSE, train_MAE))
    

# Use cross validate for everything but the matrix factorization
test_rmse, train_rmse, test_mae, train_mae = cross_validate(load_data(location), 5)

# Use this to run matrix factorization
X = reform_matrix(load_data(location))
cross_validate_matrix(X)

