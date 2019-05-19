import configparser
import sys
sys.path.append('../')
sys.path.append('../Population/')
sys.path.append('../Engine/')

config = configparser.ConfigParser()
config.read('./Experiments/config.txt')

from individual import individual
from population import population
from StateControl import decoder
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

path = "./logs/Generation27.dat"
candidation = "24"

file = open(path)
file = json.loads(file.read())
ind = file["candidates"+candidation]
ind = ind.replace('[', '').replace(']', '').replace('\n', '')
ind = ind.split(' ')
while '' in ind:
    ind.remove('')
ind = np.array(ind, dtype=np.float)
ind = ind.reshape(1, -1)

# create individual
IND = individual(config)
IND.set_dec(ind[0][0:-2])
IND.set_fitness(ind[0][-2:])
# init engin
decode = decoder(config)

# parser code
model = decode.get_model(IND.get_dec())

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./Dataset/cafir10/', train=True,
                                                download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=96,
                                                       shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./Dataset/cafir10/', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=96,
                                                      shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)  

# train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

for epoch in range(200):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# test accuracy
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(self.testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

printr('ERROR is : {0}'.format(1-(correct/total)))