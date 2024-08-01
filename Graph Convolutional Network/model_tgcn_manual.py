import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Define the calculate_laplacian_with_self_loop function
def calculate_laplacian_with_self_loop(adj):
    """
    Calculate the graph Laplacian with self-loops.

    Parameters:
    - adj: Tensor, adjacency matrix of the graph

    Returns:
    - laplacian: Tensor, graph Laplacian with self-loops
    """
    adj = adj.float()
    num_nodes = adj.size(0)
    adj_with_self_loops = adj + torch.eye(num_nodes)
    degree = torch.sum(adj_with_self_loops, dim=1)
    degree_inv_sqrt = torch.diag(degree.pow(-0.5))
    laplacian = torch.mm(torch.mm(degree_inv_sqrt, adj_with_self_loops), degree_inv_sqrt)
    return laplacian

# Load CSV data
train_data = pd.read_csv('./高邮/actual test/train data/weekday_train_data.csv')
test_data = pd.read_csv('./高邮/actual test/test data/weekday_test_data.csv')

# Convert time column to an index with specified format
train_data['time'] = pd.to_datetime(train_data['time'], format='%H:%M:%S')
test_data['time'] = pd.to_datetime(test_data['time'], format='%H:%M:%S')
train_data.set_index('time', inplace=True)
test_data.set_index('time', inplace=True)

# Normalize the data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
train_features = train_data[['total_up', 'total_down']]
train_target = train_data['total_toll'].values.reshape(-1, 1)
test_features = test_data[['total_up', 'total_down']]
test_target = test_data['total_toll'].values.reshape(-1, 1)
features_scaled = feature_scaler.fit_transform(train_features)
target_scaled = target_scaler.fit_transform(train_target)
test_features_scaled = feature_scaler.transform(test_features)
test_target_scaled = target_scaler.transform(test_target)

# Create edge index
def create_edge_index(num_nodes):
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create adjacency matrix
def create_adj_matrix(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for edge in edge_index.t().tolist():
        adj[edge[0], edge[1]] = 1
    return adj

train_edge_index = create_edge_index(len(train_data))
test_edge_index = create_edge_index(len(test_data))

train_adj_matrix = create_adj_matrix(train_edge_index, len(train_data))
test_adj_matrix = create_adj_matrix(test_edge_index, len(test_data))

# Create node features and target tensors
x = torch.tensor(features_scaled, dtype=torch.float)
y = torch.tensor(target_scaled, dtype=torch.float).view(-1, 1)
test_x = torch.tensor(test_features_scaled, dtype=torch.float)
test_y = torch.tensor(test_target_scaled, dtype=torch.float).view(-1, 1)

# Define your TGCN model
class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(adj))
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + 1) * batch_size))
        a_times_concat = self.laplacian @ concatenation
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + 1, batch_size))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", adj)
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int):
        super(TGCN, self).__init__()
        self._input_dim = adj.size(0)
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", adj)
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

# Initialize and train the TGCN model
model = TGCN(adj=train_adj_matrix, hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x.unsqueeze(0))  # Add batch dimension
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

num_epochs = 200
train_losses = []
for epoch in range(num_epochs):
    loss = train()
    train_losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Evaluation
def evaluate():
    model.eval()
    with torch.no_grad():
        prediction = model(test_x.unsqueeze(0)).view(-1).numpy()
        predictions = target_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
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

evaluate()
