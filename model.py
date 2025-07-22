import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset


class TwoLayerNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=10, output_size=10, lr=0.001, weight_decay=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.train_losses = []
        self.val_accuracies = []

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_model(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for images, labels in train_loader:
                outputs = self(images)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.train_losses.append(total_loss)
            val_accuracy = self.evaluate(val_loader, verbose=False)
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Val Accuracy: {val_accuracy*100:.2f}%")

    def evaluate(self, loader, verbose=True):
        self.eval()
        correct = total = 0

        with torch.no_grad():
            for images, labels in loader:
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if verbose:
            print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


def load_mnist_from_csv(train_csv, test_csv):
    # Load CSV data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Normalize pixel values and convert labels
    X_train = train_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    Y_train = train_df.iloc[:, 0].to_numpy(dtype=np.int64)

    X_test = test_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    Y_test = test_df.iloc[:, 0].to_numpy(dtype=np.int64)

    # Create tensor datasets
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    # Split training into train/val sets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = random_split(train_data, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    if __name__=="__main__":
        train_loader, val_loader, test_loader = load_mnist_from_csv("/path/to/train_csv_file", "/path/to/test_csv_file")
    
    fc_object = TwoLayerNN()
    fc_object.train_model(train_loader, val_loader)
    fc_object.plot_metrics()
    
    return train_loader, val_loader, test_loader
