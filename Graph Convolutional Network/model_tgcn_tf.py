import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.layers import Lambda, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Load the training CSV file
train_file_path = './高邮/actual test/train data/total_train_data.csv'
df_train = pd.read_csv(train_file_path)

# Convert the 'time' column to datetime
df_train['time'] = pd.to_datetime(df_train['time'], format='%H:%M:%S')

# Extract the hour from the time column as an integer feature
df_train['hour'] = df_train['time'].dt.hour

# Extract the relevant columns (including the hour as a feature)
features_train = df_train[['hour', 'total_up', 'total_down', 'total_toll']].values

# Create a 3D array for node features
X_train_data = []
for i in range(len(features_train)):
    hour_feature = features_train[i, 0]
    up_feature = features_train[i, 1]
    down_feature = features_train[i, 2]
    toll_feature = features_train[i, 3]
    X_train_data.append([
        [hour_feature, up_feature],
        [hour_feature, down_feature],
        [hour_feature, toll_feature]
    ])
X_train_data = np.array(X_train_data)

# Define adjacency matrix (A)
A = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, 0]
])

# Define window size, number of previous time steps used to predict the future value
window_size = 3

# Prepare input sequences and targets for training
X_train_input = []
y_train_target = []

for i in range(len(X_train_data) - window_size):
    X_train_input.append(X_train_data[i:i + window_size])
    y_train_target.append(X_train_data[i + window_size, 2, 1])  # We only want to predict total_toll (node 2, feature 1)

X_train_input = np.array(X_train_input)
y_train_target = np.array(y_train_target)

# Load the test CSV file
test_file_path = './高邮/actual test/test data/total_test_data.csv'
df_test = pd.read_csv(test_file_path)

# Convert the 'time' column to datetime
df_test['time'] = pd.to_datetime(df_test['time'], format='%H:%M:%S')

# Extract the hour from the time column as an integer feature
df_test['hour'] = df_test['time'].dt.hour

# Extract the relevant columns (including the hour as a feature)
features_test = df_test[['hour', 'total_up', 'total_down', 'total_toll']].values

# Create a 3D array for node features
X_test_data = []
for i in range(len(features_test)):
    hour_feature = features_test[i, 0]
    up_feature = features_test[i, 1]
    down_feature = features_test[i, 2]
    toll_feature = features_test[i, 3]
    X_test_data.append([
        [hour_feature, up_feature],
        [hour_feature, down_feature],
        [hour_feature, toll_feature]
    ])
X_test_data = np.array(X_test_data)

# Prepare input sequences and targets for testing
X_test_input = []
y_test_target = []

for i in range(len(X_test_data) - window_size):
    X_test_input.append(X_test_data[i:i + window_size])
    y_test_target.append(X_test_data[i + window_size, 2, 1])  # We only want to predict total_toll (node 2, feature 1)

X_test_input = np.array(X_test_input)
y_test_target = np.array(y_test_target)

# Define the T-GCN model
def build_tgcn_model(adj_matrix, input_shape):

    # Converts the adjacency matrix to a TensorFlow constant of type float32.
    A = tf.constant(adj_matrix, dtype=tf.float32)
    
    def graph_convolution(x, A):
        return tf.matmul(A, x)
    
    # Creates an input layer for the model with the specified shape.
    inputs = Input(shape=input_shape)

    # Applies the graph_convolution function to the inputs using a Lambda layer.
    gcn_layer = Lambda(lambda x: graph_convolution(x, A))(inputs)

    # Reshapes the output of the GCN layer to flatten the nodes and features.
    gcn_layer = Reshape((input_shape[0], input_shape[1] * input_shape[2]))(gcn_layer)

    # Adds a GRU (Gated Recurrent Unit) layer with 64 units. 
    # The return_sequences=True parameter ensures that the output at each time step is returned.
    gru_layer = GRU(64, return_sequences=True)(gcn_layer)  

    # Adds another GRU layer with 64 units, but this time without returning sequences.  
    gru_layer = GRU(64)(gru_layer)

    # Adds a Dense layer with 1 unit to predict a single value, total_toll.
    outputs = Dense(1)(gru_layer) 
    
    # Creates a Keras Model with the specified inputs and outputs.
    model = Model(inputs=inputs, outputs=outputs)

    # Compiles the model using the Adam optimizer and mean squared error (MSE) loss function.
    model.compile(optimizer='adam', loss='mse')
    return model

input_shape = (window_size, 3, 2)  # (time_steps, nodes, features)
tgcn_model = build_tgcn_model(A, input_shape)
tgcn_model.summary()

# Train the model and store the history
history = tgcn_model.fit(X_train_input, y_train_target, epochs=50, batch_size=16, validation_split=0.2)

# Plot the learning process
plt.figure(figsize=(12, 6))
# It measures how well the model is performing on the data it is currently being trained on. 
# A low training loss indicates that the model is fitting the training data well.
plt.plot(history.history['loss'], label='Training Loss')
# It measures how well the model generalizes to new, unseen data. 
# A low validation loss suggests that the model is generalizing well.
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.savefig('./Graph Convolutional Network/total_learning_process')
plt.show()

"""
for aj M of 200 & feature of 2000 ^
    aj M of 150 & feature of 3000 #:
Learning rate: 0.001
Batch size: 64
Training epochs: 3000
Number of hidden units: [8, 16, 32, ^64, #100, 128]
"""

# Predict on the test set
y_pred = tgcn_model.predict(X_test_input)

# Plot the comparison between actual and predicted values across all data points
plt.figure(figsize=(18, 8))

plt.plot(y_test_target, label='Actual', marker='o', linestyle='-')
plt.plot(y_pred, label='Predicted', marker='x', linestyle='--')
plt.xlabel('Data Point Index')
plt.ylabel('Total Toll')
plt.title('Comparison of Actual vs Predicted Total Toll Values')
plt.legend()
plt.tight_layout()
plt.savefig('./Graph Convolutional Network/total_prediction')
plt.show()

# Create a DataFrame to store the actual and predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test_target.flatten(),   # Flatten to ensure it's a 1D array
    'Predicted': y_pred.flatten()        # Flatten to ensure it's a 1D array
})

# Save the DataFrame to a CSV file
output_file_path = './Graph Convolutional Network/total_actual_vs_predicted.csv'
comparison_df.to_csv(output_file_path, index=False)

# Function to add value labels on bars
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart."""
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                       # Use `label` as label
            (x_value, y_value),          # Place label at end of the bar
            xytext=(0, space),           # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                 # Horizontally center label
            va=va)                       # Vertically align label


# Calculate metrics for each hour
rmse_list = []  # Root Mean Squared Error
mae_list = []  # Mean Absolute Error
accuracy_list = []
r2_list = []  # R-squared
explained_variance_list = []

for hour in range(24):
    actual = y_test_target[(hour - 3) % 24::24]
    predicted = y_pred[(hour - 3) % 24::24]
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    accuracy = 1 - (np.linalg.norm(actual - predicted) / np.linalg.norm(actual))
    r2 = r2_score(actual, predicted)
    explained_variance = explained_variance_score(actual, predicted)
    
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    accuracy_list.append(accuracy)
    explained_variance_list.append(explained_variance)

# Plot the evaluation metrics
hours = range(24)

plt.figure(figsize=(18, 15))

# Plot RMSE by hour
ax1 = plt.subplot(5, 1, 1)
ax1.bar(hours, rmse_list)
add_value_labels(ax1)
plt.xlabel('Hour')
plt.ylabel('RMSE')
plt.title('RMSE by Hour')

# Plot MAE by hour
ax2 = plt.subplot(5, 1, 2)
ax2.bar(hours, mae_list)
add_value_labels(ax2)
plt.xlabel('Hour')
plt.ylabel('MAE')
plt.title('MAE by Hour')

# Plot R-squared by hour
ax3 = plt.subplot(5, 1, 3)
ax3.bar(hours, r2_list)
add_value_labels(ax3)
plt.xlabel('Hour')
plt.ylabel('R-squared')
plt.title('R-squared by Hour')

# Plot Accuracy by hour
ax4 = plt.subplot(5, 1, 4)
ax4.bar(hours, accuracy_list)
add_value_labels(ax4)
plt.xlabel('Hour')
plt.ylabel('Accuracy')
plt.title('Accuracy by Hour')

# Plot Explained Variance by hour
ax5 = plt.subplot(5, 1, 5)
ax5.bar(hours, explained_variance_list)
add_value_labels(ax5)
plt.xlabel('Hour')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Hour')

plt.tight_layout()
plt.savefig('./Graph Convolutional Network/total_results')
plt.show()
