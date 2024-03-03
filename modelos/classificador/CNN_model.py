import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 187, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
def train(
    X_train_tensor:np.ndarray,
    y_train_tensor:np.ndarray,
    X_val_tensor:np.ndarray,
    y_val_tensor:np.ndarray,
    num_classes:int, 
    num_epochs:int=10,
    batch_size:int=50,
    lr:float=0.001,
    patience:int=5
) -> CNN:
    
    model = CNN(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    no_improvement_count = 0
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss}, Val Loss: {val_loss}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping! No improvement in validation loss.")
                break

    return model

def test(
    model: CNN,
    X_test_tensor:np.ndarray,
    y_test_tensor:np.ndarray
):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        print(classification_report(y_test_tensor, predicted))
    