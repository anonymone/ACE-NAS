import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision可以帮助我们处理常用数据集，如MNIST，COCO, ImageNET等
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import scipy.fftpack as sci
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


class Net(nn.Module):
    def __init__(self, InputDim, HiddenNum, OutputDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputDim, HiddenNum)
        self.fc1.weight.requires_grad_(requires_grad=False)
        self.fc2 = nn.Linear(HiddenNum, OutputDim,bias=False)
        self.fc2.weight.requires_grad_(requires_grad=False)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        # X = torch.sigmoid(self.fc1(X))
        X1 = X
        self.fc2.weight.data = self.fc1.weight.data.t()
        X = torch.sigmoid(self.fc2(X))
        return X, X1

    def Initialization(self, weights):
        # self.fc1.weight.data = weights.data # weights 利用Tensor 创立 其 require_grads =  False  ,不能直接赋给  fc1.weights
        # self.fc2.weight.data = weights.t().data
        self.fc1.weight.data = weights
        self.fc2.weight.data = weights.t()
    def Reset(self):
        self.fc2.weight.data = self.fc1.weight.data.t()

    def get_weights(self):
        weights_bias = torch.cat((self.fc1.weight.data.t(), self.fc1.bias.data.reshape(1, -1)))
        return weights_bias


def LoadData(batch = 64):
    # MNIST dataset
    train_dataset = dsets.MNIST(root='Data/MNIST_data',  # 选择数据的根目录
                                train=True,  # 选择训练集
                                transform=transforms.ToTensor(),  # 转换成tensor变量
                                download=False)  # 不从网络上download图片
    test_dataset = dsets.MNIST(root='Data/MNIST_data',  # 选择数据的根目录
                               train=False,  # 选择训练集
                               transform=transforms.ToTensor(),  # 转换成tensor变量
                               download=False)  # 不从网络上download图片
    T_Dim = np.array(train_dataset.train_data.shape)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True)  # 将数据打乱
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch,
                                              shuffle=True)  # 将数据打乱
    Dim = T_Dim[1] * T_Dim[2]

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    return Dim, train_loader, test_loader


def Initialization_Pop(PopSize, Dim, HiddenNum):
    # Population = (np.random.random((PopSize, Dim * HiddenNum)) - 0.5) * 2 * ((np.power(6 / (Dim + HiddenNum), 1 / 2)))
    Population = (np.random.random((PopSize, Dim * HiddenNum)) - 0.5) * 2 * ((6/np.power((Dim + HiddenNum), 1 / 2)))
    for i in range(PopSize):
        Population[i] = Population[i]*(np.random.rand( Dim * HiddenNum,) < ((i+1)/PopSize)/2)
    Boundary = np.tile([[1], [-1]], [1, Dim * HiddenNum])
    Coding = 'Real'
    return Population, Boundary, Coding


def Evaluation(Population, Dim, HiddenNum, Data):
    # Here Dim is  28*28  not 28
    pop_size = Population.shape[0]
    Weight_Grad = np.zeros(Population.shape)

    FunctionValue = np.zeros((pop_size, 2))
    # 计算 sparsity
    FunctionValue[:, 0] = np.sum(Population != 0, axis=1) / (Dim * HiddenNum)

    # Struct AE Model
    Model = Net(Dim, HiddenNum, Dim).cuda()
    # Load Train Data
    data_iter = iter(Data)
    images, labels = data_iter.next()
    images = images.view(-1, Dim).cuda()  # Dim is 28*28 not 28
    labels = labels.cuda()

    # To Vadiate the
    # im = (images[1000].numpy())
    # plt.imshow(np.reshape(im,(28,28)))
    # plt.show()

    # Define Loss funcion , here MSE is adopted
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    # Compute MSE and Grad for each weight set
    for i in range(pop_size):
        weights = torch.Tensor(np.reshape(Population[i, :], (Dim, HiddenNum))).t()
        weights.requires_grad_(requires_grad=True)
        # 修改 梯度 torch.tensor() always copies data. If you have a Tensor data and just want to change its requires_grad flag, use requires_grad_() or detach()
        # to avoid a copy. If you have a numpy array and want to avoid a copy, use torch.as_tensor(). https://pytorch.org/docs/stable/tensors.html
        Model.Initialization(weights.cuda())
        outputs = Model(images)  # here outputs contain two parts, which are final outputs and Hidden outputs
        loss = criterion(outputs[0], images)
        FunctionValue[i, 1] = loss.cpu().detach().numpy()  # loss.cpu().data.numpy()

        # FunctionValue[i, 1] = loss.detach().numpy()# here Loss is Variable
        # FunctionValue[i, 1] = loss.data.numpy()

        # Acc = SVM_Predict(images, labels, images, labels, weights)

        optimizer.zero_grad()
        loss.backward()
        Weight_Grad_Temp= np.reshape(Model.fc1.weight.grad.t().cpu().numpy(), (Dim * HiddenNum,))
        Weight_Grad_T_ = np.reshape(Model.fc2.weight.grad.cpu().numpy(), (Dim * HiddenNum,))
        Weight_Grad[i,:] = coperate_Weight_Grad(Weight_Grad_Temp,Weight_Grad_T_)

        # print(Model.fc1.weight.grad)

    return FunctionValue, Weight_Grad

def coperate_Weight_Grad(Weight_Grad_Temp,Weight_Grad_T_):
    Temp_1 = Weight_Grad_Temp.copy()
    Temp_1[Temp_1>0] = 1
    Temp_1[Temp_1 < 0] = -1
    Temp_2 = Weight_Grad_T_.copy()
    Temp_2[Temp_2 > 0] = 1
    Temp_2[Temp_2 < 0] = -1
    zeroIndex = (Temp_1 + Temp_2) == 0


    Temp_2 = Weight_Grad_T_.copy()
    prob = torch.rand(Weight_Grad_Temp.shape)>0.5
    Weight_Grad_Temp[prob] = Temp_2[prob]
    Weight_Grad_Temp[zeroIndex] = 0
    return Weight_Grad_Temp



def SVM_Predict(svm_train_data, svm_train_label, svm_test_data, svm_test_label, weights=None):  # weights 为 Hidden * Dim
    # all parameters except for  weights ，have been handled by .cuda(), so they have to convert to numpy before put into sklearn SVC
    if weights is None:
        train_data = svm_train_data
        test_data = svm_test_data
    else:
        HiddenNum, Dim = weights.shape
        Model = Net(Dim, HiddenNum, Dim).cuda()
        Model.Initialization(weights.cuda())
        _, train_data = Model(svm_train_data)
        _, test_data = Model(svm_test_data)

    # convert  data from  .cuda() to numpy
    train_data = train_data.cpu().detach().numpy()
    test_data = test_data.cpu().detach().numpy()
    train_labels = svm_train_label.cpu().detach().numpy()
    test_labels = svm_test_label.cpu().detach().numpy()
    # trian SVM model with train data
    SVM_Model = SVC(kernel='poly')
    SVM_Model.fit(train_data, train_labels)
    # print(SVM_Model.score(train_data,train_labels))

    # get accuracy for test data
    Accuracy = SVM_Model.score(test_data, test_labels)
    print(Accuracy)
    return Accuracy


class autoencoder_softmax(nn.Module):
    def __init__(self, weights, classNum):
        super(autoencoder_softmax, self).__init__()
        self.layer_num = len(weights)
        self.feature_encoder = nn.Sequential()
        # Construct n layers autoencoder
        for i in range(self.layer_num):
            self.feature_encoder.add_module("layer" + str(i), nn.Linear(weights[i].shape[0] - 1, weights[i].shape[1]))
            self.feature_encoder[2 * i].weight.data = weights[i][:-1].t()
            self.feature_encoder[2 * i].bias.data = weights[i][-1]
            self.feature_encoder.add_module("layer" + str(i) + "ActiveFunction", nn.ReLU())
            # self.feature_encoder.add_module("layer" + str(i) + "ActiveFunction", nn.Sigmoid())
        # Construct Softmax classification layer
        # self.Softmax_layer =  nn.Sequential(nn.Linear(weights[i].shape[1],classNum), nn.Softmax(dim=1))
        self.Softmax_layer = nn.Sequential(nn.Linear(weights[i].shape[1], classNum), nn.LogSoftmax(dim=1))

    def forward(self, x):
        feature = self.feature_encoder(x)
        out = self.Softmax_layer(feature)
        return out, feature


def train_layer_wise_autoencoder(Dim, HiddenNum, trainloader, Model_trans, Gene):
    Model = Net(Dim, HiddenNum, Dim).cuda()
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(Model.parameters(), lr=0.1, momentum=0.5)

    for epoch in range(Gene):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()
            if Model_trans is not None:
                _, inputs = Model_trans(inputs)
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = Model(inputs)
            loss = Loss(outputs, inputs)
            loss.backward()
            optimizer.step()
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, loss))

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    return Model.get_weights()


def train_autoencoder_n_layer_softmax():  # train_data, train_label, test_data, test_label,weights

    Dim, trainloader, testloader = LoadData(batch=50)
    weights = []
    Loadweights = np.loadtxt('W.txt')
    weights.append(torch.Tensor(Loadweights))
    # Model = None
    # layer_1_weights = train_layer_wise_autoencoder(Dim, 200, trainloader, Model, Gene=2)
    # # np.savetxt('torch_weight.txt',layer_1_weights,delimiter=' ')
    # weights.append(layer_1_weights.cpu().data)
    #
    # # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # # layer_2_weights = train_layer_wise_autoencoder(200, 100, trainloader,Model,Gene=10)
    # #
    # # weights.append(layer_2_weights.cpu().data)
    #
    # del Model
    #
    # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # layer_3_weights = train_layer_wise_autoencoder(320, 240, trainloader, Model, Gene=10)
    #
    # weights.append(layer_3_weights.cpu().data)
    #
    # del Model
    #
    # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # layer_4_weights = train_layer_wise_autoencoder(240, 120, trainloader, Model, Gene=10)
    #
    # weights.append(layer_4_weights.cpu().data)
    #
    # del Model

    # weights.append( torch.Tensor(np.random.rand(Dim+1,200) ).cuda() )#*np.random.randint(0,2,(Dim,100))
    # weights.append(torch.Tensor(np.random.rand(200+1, 200)).cuda())
    # weights.append(torch.Tensor(np.random.rand(320+1, 240)).cuda())
    # weights.append(torch.Tensor(np.random.rand(240+1, 120)).cuda())

    Model = autoencoder_softmax(weights, 10).cuda()
    # Loss = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss
    ## 多分类用的交叉熵损失函数，用这个 loss 前面不需要加 Softmax 层。
    Loss = nn.NLLLoss()
    # optimizer = torch.optim.Adam(Model.parameters(),lr=0.01)

    optimizer = torch.optim.SGD(Model.parameters(), lr=0.1, momentum=0.5)
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = Model(inputs)
            # loss = F.nll_loss(outputs, labels)
            loss = Loss(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()

            outputs, _ = Model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().data == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.5f %%' % (
            100 * correct / total))

    return None
# #
train_autoencoder_n_layer_softmax()
print("hello world!!")
