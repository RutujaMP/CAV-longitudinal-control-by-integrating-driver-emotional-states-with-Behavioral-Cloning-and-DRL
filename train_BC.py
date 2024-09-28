import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# Define the Behavioral Cloning Model
class BehavioralCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BehavioralCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Dataset class
class NGSIMDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.sort_values(by=['Vehicle_ID', 'Frame_ID'], inplace=True)
        
        # Exclude Relative_Position_Y from inputs
        self.features = self.data[['v_Vel', 'Relative_Position_X', 'Space_Headway']].values
        self.targets = self.data[['v_Acc', 'v_Vel', 'Space_Headway']].values
        
        # Normalize the features and targets
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state = self.features[idx]
        action = self.targets[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)

# Load the data
dataset = NGSIMDataset('C:/automatic_vehicular_control/datasets/processed_ngsim.csv')

# Split the dataset into training and validation sets
print("splitting the training and val sets ")
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.data['Vehicle_ID'])

train_data = torch.utils.data.Subset(dataset, train_indices)
val_data = torch.utils.data.Subset(dataset, val_indices)

# Create DataLoaders
print("train and val loading..")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Train the Behavioral Cloning Model
def train_behavioral_cloning_model(model, train_loader, val_loader, save_path, epochs=10, lr=1e-3):
    print("Inside training function...")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'Starting epoch {epoch+1}/{epochs}...')
        
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
        for states, actions in train_loader_tqdm:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader_tqdm.set_postfix({'Batch Loss': loss.item()})
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for states, actions in val_loader:
                outputs = model(states)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Time: {epoch_duration:.2f} seconds')
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)

# Specify the path to save the trained model
model_save_path = 'C:/automatic_vehicular_control/models/behavioral_cloning_model.pth'

# Instantiate and train the model
model = BehavioralCloningModel(input_dim=3, output_dim=3)  # 3 features as input, 3 target features as output
train_behavioral_cloning_model(model, train_loader, val_loader, save_path=model_save_path)

# Plotting function to visualize predictions
def plot_predictions(model, data_loader, target_scaler, num_points=100):
    print("plotting....")

    model.eval()
    states, true_actions = next(iter(data_loader))
    with torch.no_grad():
        predicted_actions = model(states)
    
    true_actions = target_scaler.inverse_transform(true_actions.numpy())
    predicted_actions = target_scaler.inverse_transform(predicted_actions.numpy())
    
    plt.figure(figsize=(10, 6))
    
    for i, label in enumerate(['Acceleration', 'Speed', 'Headway']):
        plt.subplot(3, 1, i+1)
        plt.plot(true_actions[:num_points, i], label=f'True {label}', marker='o')
        plt.plot(predicted_actions[:num_points, i], label=f'Predicted {label}', marker='x')
        plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel(label)
        plt.title(f'True vs Predicted {label}')
    
    plt.tight_layout()
    plt.show()

# Plot predictions on the validation set
plot_predictions(model, val_loader, dataset.target_scaler)





























































# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm  # Import tqdm for progress bars
# import time  # Import time module to measure the duration of each epoch

# # Define the Behavioral Cloning Model
# class BehavioralCloningModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(BehavioralCloningModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Define the Dataset class
# class NGSIMDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
#         self.data.sort_values(by=['Vehicle_ID', 'Frame_ID'], inplace=True)
        
#         self.features = self.data[['v_Vel', 'Relative_Position_X', 'Relative_Position_Y', 'Space_Headway']].values
#         self.targets = self.data[['v_Acc']].values
        
#         # Normalize the features and targets
#         self.feature_scaler = StandardScaler()
#         self.target_scaler = StandardScaler()
        
#         self.features = self.feature_scaler.fit_transform(self.features)
#         self.targets = self.target_scaler.fit_transform(self.targets)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         state = self.features[idx]
#         action = self.targets[idx]
#         return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)

# # Load the data
# dataset = NGSIMDataset('C:/automatic_vehicular_control/datasets/processed_ngsim.csv')

# # Split the dataset into training and validation sets
# print("splitting the training and val sets ")
# train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.data['Vehicle_ID'])

# train_data = torch.utils.data.Subset(dataset, train_indices)
# val_data = torch.utils.data.Subset(dataset, val_indices)

# # Create DataLoaders
# print("train and val loading..")
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# # Train the Behavioral Cloning Model
# def train_behavioral_cloning_model(model, train_loader, val_loader, save_path, epochs=10, lr=1e-3):
#     print("Inside training function...")

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         print(f'Starting epoch {epoch+1}/{epochs}...')
        
#         model.train()
#         train_loss = 0
#         train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
#         for states, actions in train_loader_tqdm:
#             optimizer.zero_grad()
#             outputs = model(states)
#             loss = criterion(outputs, actions)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             train_loader_tqdm.set_postfix({'Batch Loss': loss.item()})
        
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for states, actions in val_loader:
#                 outputs = model(states)
#                 loss = criterion(outputs, actions)
#                 val_loss += loss.item()
        
#         epoch_end_time = time.time()
#         epoch_duration = epoch_end_time - epoch_start_time
#         print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Time: {epoch_duration:.2f} seconds')
    
#     # Save the trained model
#     torch.save(model.state_dict(), save_path)

# # Specify the path to save the trained model
# model_save_path = 'C:/automatic_vehicular_control/models/behavioral_cloning_model.pth'

# # Instantiate and train the model
# model = BehavioralCloningModel(input_dim=4, output_dim=1)  # 4 features as input, 1 target feature as output
# train_behavioral_cloning_model(model, train_loader, val_loader, save_path=model_save_path)

# # Plotting function to visualize predictions
# def plot_predictions(model, data_loader, target_scaler, num_points=100):
#     print("plotting....")

#     model.eval()
#     states, true_actions = next(iter(data_loader))
#     with torch.no_grad():
#         predicted_actions = model(states)
    
#     true_actions = target_scaler.inverse_transform(true_actions.numpy())
#     predicted_actions = target_scaler.inverse_transform(predicted_actions.numpy())
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(true_actions[:num_points], label='True Actions', marker='o')
#     plt.plot(predicted_actions[:num_points], label='Predicted Actions', marker='x')
#     plt.legend()
#     plt.xlabel('Sample Index')
#     plt.ylabel('Acceleration')
#     plt.title('True vs Predicted Accelerations')
#     plt.show()

# # Plot predictions on the validation set
# plot_predictions(model, val_loader, dataset.target_scaler)
