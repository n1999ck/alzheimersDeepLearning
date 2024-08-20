import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out

dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
testDataset = dataset[int((len(dataset) * (7/8))):]
dataset = dataset[:int((len(dataset) * (7/8)))]
print(len(testDataset))
print(len(dataset))
dataset.head()
dataset_features = dataset.copy()
dataset_labels = dataset_features.pop('Diagnosis')
dataset_features = np.array(dataset_features)
print(dataset_features)
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(dataset_features) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)
input_dim = 35
output_dim = 2
hidden_dim = 100
dataset_loader =torch.utils.data.DataLoader(dataset=dataset_features,
                                            batch_size = batch_size,
                                            shuffle = True)

test_loader =torch.utils.data.DataLoader(dataset=testDataset,
                                            batch_size = batch_size,
                                            shuffle = True)

print(dataset_loader)
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
print(model)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())
print(len(list(model.parameters())))

print(list(model.parameters())[0].size())
print(list(model.parameters())[1].size())

iter = 0
for epoch in range(num_epochs):
    for i, (fields, labels) in enumerate(dataset_loader):
        optimizer.zero_grad()
        outputs = model(fields)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for fields, labels in test_loader:
                
                # Forward pass only to get logits/output
                outputs = model(fields)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
        

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))