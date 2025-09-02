import torch
import os
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

import numpy as np

import librosa
import matplotlib.pyplot as plt

SR = 48000
LR = 0.001
decay = 0.005
momentum = 0.9
nClasses = 14
batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpectrogramDataset(Dataset):
    def __init__(self, labels, audioDir, transform=None):
        self.labels = pd.read_csv(labels, header = None)
        self.audioDir = audioDir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # print(self.labels.iloc[idx])
        # print(self.labels.iloc[idx][2] + '_' + str(self.labels.iloc[idx][1]) + '.npy')
        # sample_path = os.path.join(self.audioDir, self.labels.iloc[idx][2] + '_' + str(self.labels.iloc[idx][0]) + '.npy')
        sample_path = os.path.join(self.audioDir, str(self.labels.iloc[idx][0]) + '.npy')
        s = np.load(sample_path).astype('float32')
        label = self.labels.iloc[idx][1]
        if self.transform:
            s = self.transform(s)

        return torch.from_numpy(s).reshape((1, len(s), len(s[0]))), label

class ConvTest(nn.Module):
    def __init__(self, num_classes):
        super(ConvTest, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1845888, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def load_data(labelPath, dataPath):
    ds = SpectrogramDataset(labelPath, dataPath)

    train, validate, test = random_split(ds, [0.9, 0, 0.1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader

def setup_model(num_classes, device = device):
    model = ConvTest(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = LR, weight_decay = decay, momentum=momentum)

    return model, criterion, optimizer

def train(train_loader, model, criterion, optimizer, num_epochs=10):
    best_loss = 10000
    # loss = criterion()
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            print(data.size())
            data = data.to(device)
            labels = labels.to(device)

            # print(len(data))
            outputs = model(data)
            # print(len(outputs))
            loss = criterion(outputs, labels)

            if loss.item() < best_loss:
                best_loss = loss.item()
                # model.to('cpu')
                torch.save(model.net.state_dict(), f'./data/chkpts/test.pt')
                # model.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


def test(test_loader, model):
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

def main(chkpt = None):
    trainL, testL = load_data('./data/labels.csv', './data/spectrograms')
    # print(trainL.in_order)
    # exit()
    
    model, crit, opt = setup_model(nClasses)
    
    if chkpt:
        model.load_state_dict(torch.load(chkpt))
    else:
        train(trainL, model, crit, opt)

    test(testL, model)


if __name__ == "__main__":
    # load_data('./data/labels.csv', './data/spectrograms')
    print(f"using {device}")
    # main('./data/chkpts/test.pt')
    main()