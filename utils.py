import numpy as np 
import random 

def StandardScaler(data, train=[]):
    new_data = data.copy()
    if len(train) == 0:
        col_means = np.mean(data, axis=0)
        col_std = np.std(data, axis=0)
    else:
        assert len(train) == 2, "Couldn't comprehend the mean and std values from the list input"
        col_means = train[0]
        col_std = train[1]

    for i in range(len(data)):
        new_data[i] -= col_means
        new_data[i] /= col_std
    
    if len(train) == 0 :
        return new_data, col_means, col_std
    else:
        return new_data


def mse(y_actual, y_predicted):
    total_sum = 0
    assert len(y_actual) == len(y_predicted), "the lengths of actual and predicted doesn't match"
    for i in range(len(y_actual)):
        total_sum += (y_predicted[i] - y_actual[i]) ** 2
    total_sum /= len(y_actual)
    return  total_sum


def train_test_split(X, y, test_size=0.2, shuffle=True):
    test_indices = []
    train_indices = []
    for i in range(len(X)):
        if random.uniform(0, 1) < test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    if shuffle:
        test_indices = np.random.permutation(test_indices)
        train_indices = np.random.permutation(train_indices)

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def add_bias(data):
    new_data = data.copy()
    new_data = np.c_[new_data, np.ones(len(data))]
    return new_data
