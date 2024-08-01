import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load CSV data
data = pd.read_csv('./高邮/actual test/train data/weekend_train_data.csv')

# Convert time column to an index with specified format
data['time'] = pd.to_datetime(data['time'], format='mixed')  # Adjust format as needed
data.set_index('time', inplace=True)

# Normalize the data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
features = data[['total_up', 'total_down']]
target = data['total_toll'].values.reshape(-1, 1)
features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target)

# Create edge index (assuming a temporal graph)
edge_index = []
for i in range(len(data) - 1):
    edge_index.append([i, i + 1])
    edge_index.append([i + 1, i])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create node features and target tensors
x = torch.tensor(features_scaled, dtype=torch.float)
y = torch.tensor(target_scaled, dtype=torch.float).view(-1, 1)

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

# Load test CSV data
test_data = pd.read_csv('./高邮/actual test/test data/weekend_test_data.csv')

# Convert time column to an index with specified format
test_data['time'] = pd.to_datetime(test_data['time'], format='mixed')  # Adjust format as needed
test_data.set_index('time', inplace=True)

# Normalize the test data
test_features = test_data[['total_up', 'total_down']]
test_target = test_data['total_toll'].values.reshape(-1, 1)
test_features_scaled = feature_scaler.transform(test_features)
test_target_scaled = target_scaler.transform(test_target)

# Create edge index (assuming a temporal graph)
test_edge_index = []
for i in range(len(test_data) - 1):
    test_edge_index.append([i, i + 1])
    test_edge_index.append([i + 1, i])
test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()

# Create node features and target tensors
test_x = torch.tensor(test_features_scaled, dtype=torch.float)
test_y = torch.tensor(test_target_scaled, dtype=torch.float).view(-1, 1)

# Create the graph data object
test_graph_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)

# Make predictions with the trained model
model.eval()
with torch.no_grad():
    predictions_scaled = model(test_graph_data).view(-1).numpy()

# Inverse transform the scaled predictions and target
predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1)
test_target = target_scaler.inverse_transform(test_target_scaled).reshape(-1)

# Plot the actual vs predicted data
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_target, label='Actual', color='blue')
plt.plot(test_data.index, predictions, label='Predicted', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Total Toll')
plt.xlabel('Time')
plt.ylabel('Total Toll')
plt.legend()
plt.show()

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(test_target, predictions))
mae = mean_absolute_error(test_target, predictions)

print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

# Calculate metrics for each hour
rmse_per_hour = []
mae_per_hour = []

for i in range(len(test_target)):
    rmse_per_hour.append(np.sqrt(mean_squared_error([test_target[i]], [predictions[i]])))
    mae_per_hour.append(mean_absolute_error([test_target[i]], [predictions[i]]))

# Plot RMSE per hour
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, rmse_per_hour, label='RMSE')
plt.title('RMSE per Hour')
plt.xlabel('Hour')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Plot MAE per hour
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, mae_per_hour, label='MAE')
plt.title('MAE per Hour')
plt.xlabel('Hour')
plt.ylabel('MAE')
plt.legend()
plt.show()
