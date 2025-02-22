import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


# Configure logging
logging.basicConfig(filename='SmartModelDP.log', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w')  # Overwrite the log file each time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 filters
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with kernel size 2 and stride 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer (flattened input)
        self.fc2 = nn.Linear(128, 10)          # Output layer for 10 classes (digits 0-9)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))     # First fully connected layer with ReLU
        x = self.fc2(x)             # Second fully connected layer (output layer)
        return x

def CnnTrain(datasetTrain, initial_gradients, worker_id, EPOCH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataLoader = DataLoader(datasetTrain, batch_size=64, shuffle=True)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if initial_gradients:
        model.load_state_dict(initial_gradients)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Batch {batch_idx}: Loss: {loss.item():.6f}, '
                        f'Accuracy: {100. * correct / total:.2f}%')

    average_loss = total_loss / len(dataLoader)
    accuracy = 100. * correct / total
    logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Training complete. '
                f'Average loss: {average_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    
    gradients = serialize_gradients(model.state_dict())
    return gradients , accuracy , average_loss

def CnnTest(gradients , test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    
    # Load the model with the provided gradients
    if gradients:
        model.load_state_dict(gradients)
    
    model.eval()  # Set the model to evaluation mode
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if accuracy >= 50:
        SaveModel(model)
    
    return accuracy, test_loss

def serialize_gradients(state_dict):
    return {name: param.cpu().tolist() for name, param in state_dict.items()}

def deserialize_gradients(serialized_gradients):
    return {name: torch.tensor(param) for name, param in serialized_gradients.items()}

def SaveModel(model):
    torch.save(model.state_dict(), 'finalCNN.pth')
    print("--> Model saved as 'finalCNN.pth'")
    
def AverageGradients(current, new):
    average = {}
    if current is None:
        return new
    for key in current.keys():
        if key in new:
            avg_values = np.mean([current[key], new[key]], axis=0)
            average[key] = avg_values.tolist()  # Convert NumPy array back to list
    return average