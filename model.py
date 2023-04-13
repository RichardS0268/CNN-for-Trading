# each day is displayed in 3 pixels
# use cnn to predict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class CNN5d(nn.Module):
    # input in shape of N*15*32
    # output in shape of N*1
    def __init__(self):
        super(CNN5d, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class CNN20d(nn.Module):
    # input in shape of N*60*64
    # output in shape of N*1
    # with 3 conv layers
    def __init__(self):
        super(CNN20d, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu', weight_decay=0.0):
    model = model.to(device)
    # train model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.MSELoss().to(device)
    for epoch in range(num_epochs):
        # training
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
        print('| Epoch: %d | Loss: %.4f |' % (epoch, loss.item()))

        # validation
        v_loss = 0
        v_accu = 0
        model.eval()
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            v_loss += loss.item()
            v_accu += torch.sum(torch.abs(outputs - labels) < 0.5).item()
        v_loss /= len(val_loader)
        v_accu /= len(val_loader) * batch_size
        print('| Validation Loss: %.4f | Accuracy: %.4f |' % (v_loss, v_accu))

    return model

def test_model(model, test_loader, batch_size=32, device='cpu'):
    model = model.to(device)
    loss_func = nn.MSELoss().to(device)
    model.eval()
    t_loss = 0
    t_accu = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        t_loss += loss.item()
        t_accu += torch.sum(torch.abs(outputs - labels) < 0.5).item()
    t_loss /= len(test_loader)
    t_accu /= len(test_loader) * batch_size
    print('| Test Loss: %.4f | Accuracy: %.4f |' % (t_loss, t_accu))