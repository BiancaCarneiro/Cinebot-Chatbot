import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

def train_lstm(
        X_train_tensor:np.ndarray,
        y_train_tensor:np.ndarray,
        X_val_tensor:np.ndarray,
        y_val_tensor:np.ndarray,
        num_classes:int, 
        num_epochs:int=10,
        batch_size:int=50,
        lr:float=0.001,
        patience:int=5,
        num_layers:int=3,
        hidden_size:int=128
    ) -> LSTMModel:
    
    input_size = X_train_tensor.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Treinando LSTM no device {device}")
    
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.unsqueeze(1))
            val_loss = criterion(val_outputs, y_val_tensor)
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss}, Val Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping! No improvement in validation loss.")
                break    
    return model

def test_LSTM(
        model: LSTMModel,
        X_test_tensor:np.ndarray,
        y_test_tensor:np.ndarray
    ):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.unsqueeze(1)) 
        _, predicted = torch.max(outputs, 1)
        print(classification_report(y_test_tensor, predicted))