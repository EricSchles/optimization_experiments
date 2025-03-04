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
