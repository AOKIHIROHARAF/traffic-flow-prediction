import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
data = pd.read_csv('./高邮/actual test/train data/weekend_train_data.csv')

# Convert time column to an index with specified format
data['time'] = pd.to_datetime(data['time'], format='mixed')  # Adjust format as needed
data.set_index('time', inplace=True)

# Extract features and target
features = data[['total_up', 'total_down']].values
target = data['total_toll'].values

# Create edge index (assuming a temporal graph)
edge_index = []
for i in range(len(data) - 1):
    edge_index.append([i, i + 1])
    edge_index.append([i + 1, i])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create node features and target tensors
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(target, dtype=torch.float).view(-1, 1)

# Create the graph data object
graph_data = Data(x=x, edge_index=edge_index, y=y)

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
model = GCN(in_channels=2, hidden_channels=16, out_channels=1)

# Set up the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Store the loss values
train_losses = []
eval_losses = []

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = criterion(out, graph_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training for a number of epochs
num_epochs = 200
for epoch in range(num_epochs):
    loss = train()
    train_losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Evaluation
def evaluate():
    model.eval()
    with torch.no_grad():
        prediction = model(graph_data)
        loss = criterion(prediction, graph_data.y)
        eval_losses.append(loss.item())
        print(f'Evaluation Loss: {loss:.4f}')

# Evaluate the model
evaluate()

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the evaluation loss
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), eval_losses * num_epochs, label='Evaluation Loss')
plt.title('Evaluation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load test CSV data
test_data = pd.read_csv('./高邮/actual test/test data/weekend_test_data.csv')

# Convert time column to an index with specified format
test_data['time'] = pd.to_datetime(test_data['time'], format='mixed')  # Adjust format as needed
test_data.set_index('time', inplace=True)

# Extract features and target
test_features = test_data[['total_up', 'total_down']].values
test_target = test_data['total_toll'].values

# Create edge index (assuming a temporal graph)
test_edge_index = []
for i in range(len(test_data) - 1):
    test_edge_index.append([i, i + 1])
    test_edge_index.append([i + 1, i])
test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()

# Create node features and target tensors
test_x = torch.tensor(test_features, dtype=torch.float)
test_y = torch.tensor(test_target, dtype=torch.float).view(-1, 1)

# Create the graph data object
test_graph_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)

# Make predictions with the trained model
model.eval()
with torch.no_grad():
    predictions = model(test_graph_data).view(-1).numpy()

# Plot the actual vs predicted data
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_target, label='Actual', color='blue')
plt.plot(test_data.index, predictions, label='Predicted', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Total Toll')
plt.xlabel('Time')
plt.ylabel('Total Toll')
plt.legend()
plt.show()

# Calculate the difference between actual and predicted values
difference = test_target - predictions

# Print the mean absolute error
mean_absolute_error = np.mean(np.abs(difference))
print(f'Mean Absolute Error: {mean_absolute_error:.4f}')
