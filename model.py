# each day is displayed in 3 pixels
# use cnn to predict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import enum

class MODEL_INPUT(enum.Enum):
    FIVE_DAYS = 1
    TWENTY_DAYS = 2

class MODEL_OUTPUT(enum.Enum):
    ONE_DAY = 1
    FIVE_DAYS = 2

class CNN5d(nn.Module):
    # input in shape of N*32*15
    # output in shape of N*1
    def __init__(self):
        super(CNN5d, self).__init__()
        # conv1 is 5x3 conv, 64
        # conv2 is 5x3 conv, 128
        self.conv1 = nn.Conv2d(1, 64, (5, 3), padding=(2, 1))
        self.conv2 = nn.Conv2d(64, 128, (5, 3), padding=(2, 1))
        self.fc1 = nn.Linear(15360, 2)

    def forward(self, x):
        # input: N * 32 * 15
        x = x.unsqueeze(1)   # output: N * 1 * 32 * 15
        x = F.relu(self.conv1(x))   # output: N * 64 * 32 * 15
        x = F.max_pool2d(x, (2, 1)) # output: N * 64 * 16 * 15
        x = F.relu(self.conv2(x))   # output: N * 128 * 16 * 15
        x = F.max_pool2d(x, (2, 1)) # output: N * 128 * 8 * 15
        x = x.view(-1, 15360)       # output: N * 15360
        x = self.fc1(x)             # output: N * 1
        x = F.softmax(x, dim=1)     # output: N * 1
        return x
    
class CNN20d(nn.Module):
    # input in shape of N*64*60
    # output in shape of N*1
    # with 3 conv layers
    def __init__(self):
        super(CNN20d, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (5, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 128, (5, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, (5, 3), padding=1)
        self.fc1 = nn.Linear(46080, 2)

    def forward(self, x):
        # input: N * 64 * 60
        x = x.unsqueeze(1)   # output: N * 1 * 64 * 60
        print(x.shape)
        x = F.relu(self.conv1(x))   # output: N * 64 * 62 * 60
        print(x.shape)
        x = F.max_pool2d(x, (2, 1)) # output: N * 64 * 31 * 60
        print(x.shape)
        x = F.relu(self.conv2(x))   # output: N * 128 * 29 * 60
        print(x.shape)
        x = F.max_pool2d(x, (2, 1)) # output: N * 128 * 14 * 60
        print(x.shape)
        x = F.relu(self.conv3(x))   # output: N * 256 * 12 * 60
        print(x.shape)
        x = F.max_pool2d(x, (2, 1)) # output: N * 256 * 6 * 60
        print(x.shape)
        print("before view: ", x.shape)
        x = x.view(x.shape[0], -1)       # output: N * 46080
        print(x.shape)
        x = self.fc1(x)             # output: N * 2
        print(x.shape)
        x = F.softmax(x, dim=1)     # output: N * 2
        print(x.shape)
        return x

def train_model(models, train_loader, val_loader, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu', weight_decay=0.0):
    # train model
    optimizer_1 = optim.Adam(models[0].parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_5 = optim.Adam(models[1].parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_20 = optim.Adam(models[2].parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.BCELoss().to(device)
    for epoch in range(num_epochs):
        # training
        models[0].train()
        models[1].train()
        models[2].train()
        for i, (inputs, ret1, ret5, ret20) in enumerate(train_loader):
            print(inputs, ret1, ret5, ret20)
            inputs = inputs.to(torch.float32).to(device)
            ret1 = ret1.to(device)
            ret5 = ret5.to(device)
            ret20 = ret20.to(device)
            optimizer_1.zero_grad()
            optimizer_5.zero_grad()
            optimizer_20.zero_grad()
            outputs_1 = models[0](inputs)
            outputs_5 = models[1](inputs)
            outputs_20 = models[2](inputs)
            print(outputs_1.shape)
            loss_1 = loss_func(outputs_1, ret1)
            loss_5 = loss_func(outputs_5, ret5)
            loss_20 = loss_func(outputs_20, ret20)
            
            loss_1.backward(retain_graph=True)
            loss_5.backward(retain_graph=True)
            loss_20.backward(retain_graph=True)
            optimizer_1.step()
            optimizer_5.step()
            optimizer_20.step()
        print('| Epoch: %d | Loss_1: %.4f | Loss_5: %.4f |Loss_20: %.4f |' % (epoch, loss_1.item(), loss_5.item(), loss_20.item()))

        # validation
        v_loss_1 = 0
        v_loss_5 = 0
        v_loss_20 = 0
        v_accu_1 = 0
        v_accu_5 = 0
        v_accu_20 = 0
        models[0].eval()
        models[1].eval()
        models[2].eval()
        for i, (inputs, ret1, ret5, ret20) in enumerate(val_loader):
            inputs = inputs.to(torch.float32).to(device)
            ret1 = ret1.to(device)
            ret5 = ret5.to(device)
            ret20 = ret20.to(device)
            outputs_1 = models[0](inputs)
            outputs_5 = models[1](inputs)
            outputs_20 = models[2](inputs)
            loss_1 = loss_func(outputs_1, ret1)
            loss_5 = loss_func(outputs_5, ret5)
            loss_20 = loss_func(outputs_20, ret20)
            v_loss_1 += loss_1.item()
            v_loss_5 += loss_5.item()
            v_loss_20 += loss_20.item()
            v_accu_1 += torch.sum(torch.abs(outputs_1 - ret1) < 0.5).item()
            v_accu_5 += torch.sum(torch.abs(outputs_5 - ret5) < 0.5).item()
            v_accu_20 += torch.sum(torch.abs(outputs_20 - ret20) < 0.5).item()
        v_loss_1 /= len(val_loader)
        v_loss_5 /= len(val_loader)
        v_loss_20 /= len(val_loader)
        v_accu_1 /= len(val_loader) * batch_size
        v_accu_5 /= len(val_loader) * batch_size
        v_accu_20 /= len(val_loader) * batch_size
        print('| Epoch: %d | Val Loss_1: %.4f | Val Loss_5: %.4f | Val Loss_20: %.4f | Val Accuracy_1: %.4f | Val Accuracy_5: %.4f | Val Accuracy_20: %.4f |' % (epoch, v_loss_1, v_loss_5, v_loss_20, v_accu_1, v_accu_5, v_accu_20))

    return models

def test_model(models, test_loader, batch_size=32, device='cpu'):
    loss_func = nn.BCELoss().to(device)
    models[0].eval()
    models[1].eval()
    models[2].eval()
    t_loss_1 = 0
    t_loss_5 = 0
    t_loss_20 = 0
    t_accu_1 = 0
    t_accu_5 = 0
    t_accu_20 = 0
    for i, (input, ret1, ret5, ret20) in enumerate(test_loader):
        inputs = inputs.to(torch.float32).to(device)
        ret1 = ret1.to(device)
        ret5 = ret5.to(device)
        ret20 = ret20.to(device)
        outputs_1 = models[0](inputs)
        outputs_5 = models[1](inputs)
        outputs_20 = models[2](inputs)
        loss_1 = loss_func(outputs_1, ret1)
        loss_5 = loss_func(outputs_5, ret5)
        loss_20 = loss_func(outputs_20, ret20)
        t_loss_1 += loss_1.item()
        t_loss_5 += loss_5.item()
        t_loss_20 += loss_20.item()
        t_accu_1 += torch.sum(torch.abs(outputs_1 - ret1) < 0.5).item()
        t_accu_5 += torch.sum(torch.abs(outputs_5 - ret5) < 0.5).item()
        t_accu_20 += torch.sum(torch.abs(outputs_20 - ret20) < 0.5).item()
    t_loss_1 /= len(test_loader)
    t_loss_5 /= len(test_loader)
    t_loss_20 /= len(test_loader)
    t_accu_1 /= len(test_loader) * batch_size
    t_accu_5 /= len(test_loader) * batch_size
    t_accu_20 /= len(test_loader) * batch_size
    print('| Test Loss_1: %.4f | Test Loss_5: %.4f | Test Loss_20: %.4f | Test Accuracy_1: %.4f | Test Accuracy_5: %.4f | Test Accuracy_20: %.4f |' % (t_loss_1, t_loss_5, t_loss_20, t_accu_1, t_accu_5, t_accu_20))