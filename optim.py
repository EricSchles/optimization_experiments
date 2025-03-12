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
from dataset_creator import gen_training_data


class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HybridModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.sklearn_classifier = LogisticRegression()
        self.fitted = False

    def forward(self, x):
        if not self.fitted:
            raise Exception("Scikit-learn classifier not fitted yet")

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        detached_x = x.detach().numpy()
        sklearn_output = self.sklearn_classifier.predict(detached_x)
        return torch.from_numpy(sklearn_output)

    def fit_sklearn_classifier(self, x, y):
        x_numpy = x.values
        self.sklearn_classifier.fit(x_numpy, y)
        self.fitted = True

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

(X_train_cov, X_test_cov, y_train_cov, y_test_cov), df_cov = gen_training_data()

X_train = X_train_cov[["Age", "Salary", "Longitude", "Latitude"]]
y_train = X_train_cov["party"]

X_test = X_test_cov[["Age", "Salary", "Longitude", "Latitude"]]
y_test = X_test_cov["party"]

model = HybridModel(4, 4, 4)
model.fit_sklearn_classifier(X_train, y_train)
# Construct our model by instantiating the class defined above

dataset = CustomDataset(df_cov)

# 4. Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    for batch_features, batch_targets in dataloader:
        with torch.enable_grad():
            y_pred = model(batch_features)
            # Compute and print loss
            loss = criterion(y_pred, batch_targets)
            
    if t % 100 == 99:
        print(t, loss.item())
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    try:
        loss.backward()
        optimizer.step()
    except:
        print(t)
        break

import code
code.interact(local=locals())
