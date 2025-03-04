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

def split_unevenly(df):
    half = len(df)/2
    if int(half) != half:
        return int(half)+1, int(half)
    return int(half), int(half)

def gen_training_data():
    df_cov = pd.DataFrame()

    df_cov["party"] = [random.choice(["republican", "democrat"])
                       for _ in range(2000)]
    democrat = df_cov[df_cov["party"] == "democrat"]
    republican = df_cov[df_cov["party"] == "republican"]
    first_half_republican, second_half_republican = split_unevenly(republican)
    first_half_democrat, second_half_democrat = split_unevenly(democrat)

    democrat["Age"] = np.random.normal(35, 19, size=len(democrat))
    democrat["Age"] = democrat["Age"].astype(int)
    republican["Age"] = np.random.normal(55, 10, size=len(republican))
    republican["Age"] = republican["Age"].astype(int)

    democrat["Salary"] = np.random.normal(65000, 15000, size=len(democrat))
    democrat["Salary"] = democrat["Salary"].apply(lambda x: round(x, 2))
    republican["Salary"] = np.concatenate([
        np.random.normal(25000, 1500, size=first_half_republican),
        np.random.normal(155000, 10000, size=second_half_republican)
    ])
    republican["Salary"] = republican["Salary"].apply(lambda x: round(x, 2))

    democrat["Latitude"] = np.concatenate([
        np.random.normal(40, 1, size=first_half_democrat),
        np.random.normal(34, 1, size=second_half_democrat)
    ])
    democrat["Latitude"] = democrat["Latitude"].apply(lambda x: round(x, 4))
    republican["Latitude"] = np.random.normal(39, 15, size=len(republican))
    republican["Latitude"] = republican["Latitude"].apply(lambda x: round(x, 4))

    democrat["Longitude"] = np.random.normal(94, 15, size=len(democrat))
    democrat["Longitude"] = democrat["Longitude"].apply(lambda x: round(x, 4))
    republican["Longitude"] = np.random.normal(94, 15, size=len(republican))
    republican["Longitude"] = republican["Longitude"].apply(lambda x: round(x, 4))

    df_cov = pd.concat([democrat, republican])
    df_cov["intelligence"] = df_cov["Salary"] * df_cov["Age"]

    df_cov["party"] = df_cov["party"].map({"republican": 0, "democrat": 1})
    X_cov = df_cov[["Age", "Salary", "Longitude", "Latitude", "party"]]
    y_cov = df_cov["intelligence"]
    return train_test_split(X_cov, y_cov, random_state=1), df_cov


class Dense:
    def __init__(self, input_dim, output_dim, activation_function):
        self.synapse = 2 * np.random.random((input_dim, output_dim)) - 1
        self.select_activation_function(activation_function)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, y):
        return 1 - y ** 2

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dsigmoid(self, y):
        return y*(1-y)

    def relu(self, x):
        return x * (x > 0)

    def drelu(self, x):
        return 1. * (x > 0)
    
    def select_activation_function(self, activation_function):
        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.dtanh
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.dsigmoid
        if activation_function == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.drelu
        if activation_function == "linear":
            self.activation_function = lambda x: x
            self.activation_derivative = lambda x: x

    def forward(self, previous_layer):
        self.output = self.activation_function(
            previous_layer.dot(self.synapse)
        )
        return self.output
                           
    def compute_gradient(self, layer, error):
        self.delta = error * self.activation_derivative(layer)
        return self.delta.dot(self.synapse.T)

    def prepare_for_multiplication(self, vector):
        num_cols = len(vector)
        num_rows = 1
        return vector.reshape(num_rows, num_cols)

    def update_weights(self, layer, learning_rate):
        layer = self.prepare_for_multiplication(layer)
        delta = self.prepare_for_multiplication(self.delta)
        self.synapse += layer.T.dot(delta) * learning_rate

class Network:
    def __init__(self, layers, fit_model):
        self.nn = layers
        self.layers = []
        self.fit_model = fit_model
        
    def forward(self, X):
        self.layers = []
        self.layers.append(X)
        for index, synapse in enumerate(self.nn):
            output = synapse.forward(self.layers[index])
            self.layers.append(output)

        feature_tensor = []
        for index, elem in enumerate(X):
            feature_tensor.append(
                synapse.synapse[index]*elem
            )
        feature_tensor = np.array(feature_tensor).T
        output = self.fit_model.predict(feature_tensor)
        return output

    def backpropagate(self, error, learning_rate):
        layers = list(reversed(self.layers))
        nn = list(reversed(self.nn))
        for index, layer in enumerate(layers[:-1]):
            error = nn[index].compute_gradient(layer, error)

        for index, synapse in enumerate(self.nn):
            synapse.update_weights(self.layers[index], learning_rate)

        
class NeuralNetwork:
    def __init__(self, fit_model, learning_rate=0.1, target_mse=0.01, epochs=2):
        self.layers = []
        self.connections = []
        self.network = None
        self.learning_rate = learning_rate
        self.target_mse = target_mse
        self.epochs = epochs
        self.errors = []
        self.fit_model = fit_model
        
    def add_layer(self, layer):
        self.layers.append(layer)

    def init_network(self):
        self.network = Network(self.layers, self.fit_model)

    def fit(self, X, y):
        self.init_network()
        for epoch in range(self.epochs):
            self.errors = []

            rows, columns = X.shape
            for index in range(rows):
                # Forward
                output = self.network.forward(X[index])
                # Compute the error
                error = y[index] - output
                self.errors.append(error)

                # Back-propagate the error
                self.network.backpropagate(error, self.learning_rate)

            mse = (np.array(self.errors) ** 2).mean()
            if mse <= self.target_mse:
                break

    def predict(self, X):
        rows, columns = X.shape
        output = []
        for index in range(rows):
            output.append(
                self.network.forward(X[index])
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
nn.add_layer(Dense(4, 4, "relu"))
nn.add_layer(Dense(4, 1, "linear"))
nn.fit(X.values, y.values)
y_pred = nn.predict(X.values)
print(mean_squared_error(y_pred, y.values))


# class HybridModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(HybridModel, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_size, num_classes)
#         self.sklearn_classifier = LogisticRegression()
#         self.fitted = False

#     def forward(self, x):
#         if not self.fitted:
#             raise Exception("Scikit-learn classifier not fitted yet")

#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)

#         detached_x = x.detach().numpy()
#         sklearn_output = self.sklearn_classifier.predict(detached_x)
#         return torch.from_numpy(sklearn_output)

#     def fit_sklearn_classifier(self, x, y):
#         x_numpy = x.values
#         self.sklearn_classifier.fit(x_numpy, y)
#         self.fitted = True

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(
            dataframe[['Age', 'Salary', "Longitude", "Latitude"]].values, dtype=torch.float32
        )
        self.targets = torch.tensor(dataframe['intelligence'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



#model = HybridModel(4, 4, 4)
#model.fit_sklearn_classifier(X_train, y_train)
# Construct our model by instantiating the class defined above

# dataset = CustomDataset(df_cov)

# # 4. Create a DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Construct our loss function and an Optimizer. The call to model.parameters()
# # in the SGD constructor will contain the learnable parameters (defined 
# # with torch.nn.Parameter) which are members of the model.
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
# for t in range(5):
#     # Forward pass: Compute predicted y by passing x to the model
#     for batch_features, batch_targets in dataloader:
#         with torch.enable_grad():
#             y_pred = model(batch_features)
#             # Compute and print loss
#             loss = criterion(y_pred, batch_targets)
            
#     if t % 100 == 99:
#         print(t, loss.item())
    
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     try:
#         loss.backward()
#         optimizer.step()
#     except:
#         print(t)
#         break

import code
code.interact(local=locals())
