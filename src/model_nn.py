import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class BasicFeedforward(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64]):
        super(BasicFeedforward, self).__init__()
        
        # Create a list of layers dynamically based on hidden_sizes
        layers = []
        current_size = input_size
        
        # Build the network architecture
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Normalize activations
                nn.ReLU(),                    # Non-linear activation
                nn.Dropout(0.3)               # Regularization
            ])
            current_size = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_neural_network():
    # Load train and test datasets
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Separate features and label
    X_train = train_df.drop(columns=["style"]).values
    y_train = train_df["style"].values
    X_test = test_df.drop(columns=["style"]).values
    y_test = test_df["style"].values
    
    # Encode the labels since they are strings
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Scale features (fit new scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    
    # Create DataLoader for batching
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    input_size = X_train_scaled.shape[1]
    num_classes = len(le.classes_)
    
    model = BasicFeedforward(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix for test set
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save confusion matrix as PNG
    plt.savefig("output/conf_nn.png", bbox_inches="tight")
    print("Confusion matrix saved as 'output/conf_nn.png'")
    
    # Save the model, scaler, and label encoder for later use
    joblib.dump(model, "models/nn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("Model, scaler, and label encoder saved.")

if __name__ == "__main__":
    train_neural_network()