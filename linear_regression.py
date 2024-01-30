import csv 
from utils import StandardScaler, train_test_split, add_bias, mse
import numpy as np 
import math
import pickle
import os

data = []
with open('Real estate.csv') as f:
    temp_data = csv.reader(f)
    for row in temp_data:
        data.append(row)

columns = data[0]
data = np.array(data[1:], dtype=np.float64)

X = data[:, 1:7]
y = data[:, 7]

X_train, y_train, X_test, y_test = train_test_split(X, y)

X_train, col_mean, col_std = StandardScaler(X_train)
X_train = add_bias(X_train)


class LinearRegression:
    def __init__(self):
        self.model = []

    # Ordinary Least Squares Method
    def Train(self, X_training, y_training):
        xt_x_inv = np.linalg.inv(np.dot(X_training.T, X_training))
        xt_y = np.dot(X_training.T, y_training)
        self.model = np.dot(xt_x_inv, xt_y)

    def predict(self, x_new):
        assert len(self.model) > 0, "use the Train function first to obtain the parameters before predicting"
        weight = self.model[:-1]
        bias = self.model[-1]
        return np.dot(x_new, weight) + bias
    def save(self, filename):
        if os.path.exists(filename):
            change = input("A file with similar name already exists do you want to overwrite ?(Y/N) - ")
            if change.lower() == 'y':
                print("Saving the model")
                with open(filename, 'wb') as f:
                    pickle.dump(self.model, f)
            else:
                return 
        else:
            print("Saving the model")
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, filename):
        assert os.path.exists(filename), "the file to be loaded doesn't exist"
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        

regressor = LinearRegression()
file_name = "lr_estate.pickle"
if os.path.exists(file_name):
    print("Trained model exists so, loading the model")
    regressor.load(file_name)
else:
    regressor.Train(X_train, y_train)
    regressor.save(file_name)

X_test = StandardScaler(X_test, train=[col_mean, col_std])
y_predicted = regressor.predict(X_test)

print(np.mean(y_train))
print(math.sqrt(mse(y_predicted, y_test))) # Root Mean-Squared Error -> output in the same unit as our data
