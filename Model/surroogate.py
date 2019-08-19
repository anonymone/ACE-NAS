import numpy as np
import torch 
import torch.nn as nn

class RankNet(nn.Module):
    def __init__(self, sizeList=[(256,128),(128,64),(64,32),(32,2)]):
        super(RankNet,self).__init__()
        self.model = nn.Sequential()
        for inSize, outSize in sizeList[:-1]:
            self.model.add_module('Linear{0}_{1}'.format(inSize,outSize),nn.Linear(inSize,outSize))
            self.model.add_module('ReLU',nn.ReLU())
        self.model.add_module('Linear{0}_{1}'.format(sizeList[-1][0],sizeList[-1][1]),nn.Linear(sizeList[-1][0],sizeList[-1][1]))
        self.P_ij = nn.Sigmoid()
    def forward(self, input1, input2):
        x_i = self.model(input1)
        x_j = self.model(input2)
        S_ij = x_i - x_j
        return self.P_ij(S_ij)
    
    def predict(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = RankNet()
    criterion = nn.NLLLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters)
    # dataset 
    data = np.random.rand(1000,257).astype(dtype="float32")
    print(data[0:10,:])

    y_i = data[0:500,-1]
    y_j = data[500:1000,-1]

    labels = []
    for y in zip(y_i,y_j):
        if y[0] >= y[1]:
            labels.append(1)
        else:
            labels.append(0)
    #    else:
    #        labels.append(0.5)
    labels = np.array(labels, dtype="int64")
    print(labels)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    outputs = model(data[0:500,:-1],data[500:1000,:-1])
    np.sum(np.argmax(outputs.data.numpy(),1) == labels.data.numpy())/500
    for i in range(200):
        optimizer.zero_grad()
        outputs = model(data[0:500,:-1],data[500:1000,:-1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(np.sum(np.argmax(model(data[0:500,:-1],data[500:1000,:-1]).data.numpy(),1) == labels.data.numpy())/500)