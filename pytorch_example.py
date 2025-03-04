import torch
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a,b,c,d = self.a*x[:,0] + self.b*x[:,1] + self.c*self.d
        tensor = [a,b,c,d]
        pred = self.fit_model.predict(tensor)
        return pred

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} * feature1 + {self.b.item()} * feature2 + {self.c.item()} * {self.d.item()}'


# 1. Create a Pandas DataFrame (replace with your data loading)
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# 2. Define a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(
            dataframe[['feature1', 'feature2']].values, dtype=torch.float32
        )
        self.targets = torch.tensor(dataframe['target'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 3. Create an instance of the Dataset
dataset = CustomDataset(df)

# 4. Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 5. Iterate through the DataLoader
    # Perform training or evaluation steps here
    
# Construct our model by instantiating the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    for batch_features, batch_targets in dataloader:
        
        print("Batch Targets:", )
        y_pred = model(batch_features)

        # Compute and print loss
        loss = criterion(y_pred, batch_targets)
        
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
