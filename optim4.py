import numpy as np 
import math
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import code
from dataset_creator import gen_training_data
from network_utils import Dense


class Network:
    def __init__(self, layers, fit_model):
        self.nn = layers
        self.layers = []
        self.fit_model = fit_model
        
    def forward(self, X, weights):
        self.layers = []
        self.layers.append(X)
        for index, synapse in enumerate(self.nn):
            output = synapse.forward(self.layers[index])
            self.layers.append(output)

        # will have to mess with this a bit
        # when going to prod.
        # x_s*m_s + x_b*m_b        
        feature_tensor = []
        for index, elem in enumerate(X):
            feature_tensor.append(
                weights[index]*elem
            )
        feature_tensor = np.array(feature_tensor).T
        output = self.fit_model.predict(feature_tensor.reshape(1, -1))
        return output

    def backpropagate(self, error, learning_rate):
        layers = list(reversed(self.layers))
        nn = list(reversed(self.nn))
        for index, layer in enumerate(layers[:-1]):
            error = nn[index].compute_gradient(layer, error)

        for index, synapse in enumerate(self.nn):
            synapse.update_weights(self.layers[index], learning_rate)

# This always produces singular matrices.
# Why?
class NeuralNetwork:
    def __init__(self, fit_model, learning_rate=0.001, target_mse=0.01, epochs=2):
        self.layers = []
        self.connections = []
        self.network = None
        self.learning_rate = learning_rate
        self.target_mse = target_mse
        self.epochs = epochs
        self.errors = []
        self.fit_model = fit_model
        self.weights = None
        
    def add_layer(self, layer):
        self.layers.append(layer)

    def init_network(self):
        self.network = Network(self.layers, self.fit_model)

    def lin_reg_fit(self, weights, X, y):
        x_transpose_times_x = weights.T.dot(weights)
        try:
            inverse = np.linalg.inv(x_transpose_times_x)
        except:
            code.interact(local=locals())
        return inverse.dot(X.T).dot(y)        

    def lin_reg_predict(self, X):
        return X.dot(self.weights)
    
    def fit(self, X, y):
        self.init_network()
        for epoch in range(self.epochs):
            self.errors = []

            rows, columns = X.shape
            for index in range(rows):
                # Forward
                self.weights = self.lin_reg_fit(
                    self.network.nn[0].synapse, X, y
                )
                output = self.network.forward(X[index], self.weights)
                print(index)
                # Compute the error
                error = y[index] - output
                self.errors.append(error)

                rmse = np.sqrt((np.array(self.errors) ** 2).mean())
                # Back-propagate the error
                self.network.backpropagate(rmse, self.learning_rate)

            mse = (np.array(self.errors) ** 2).mean()
            if mse <= self.target_mse:
                break

    def predict(self, X):
        rows, columns = X.shape
        output = []
        for index in range(rows):
            output.append(
                self.network.forward(X[index], self.weights)
            )
        return np.array(output)

(X_train_cov, X_test_cov, y_train_cov, y_test_cov), df_cov = gen_training_data()

X_train = X_train_cov[["Age", "Salary", "Longitude", "Latitude"]]
y_train = X_train_cov["party"]

X_test = X_test_cov[["Age", "Salary", "Longitude", "Latitude"]]
y_test = X_test_cov["party"]

X = df_cov[["Age", "Salary", "Longitude", "Latitude"]]
y = df_cov["intelligence"]
logit = LogisticRegression()
logit.fit(X_train, y_train)

nn = NeuralNetwork(logit)
# Dense(inputs, outputs, activation)
nn.add_layer(Dense(4, 4, "sigmoid"))
nn.fit(X.values, y.values)
y_pred = nn.predict(X.values).reshape(-1)
y_pred_lin = nn.lin_reg_predict(X.values)
print(mean_squared_error(y_pred, y.values))
print(mean_squared_error(y_pred_lin, y.values))

#folks might get annoyed - if so,
# kill backpass
# just do forward pass with weight optimization


import code
code.interact(local=locals())
