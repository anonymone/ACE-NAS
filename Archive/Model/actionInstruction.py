import torch.nn as nn
import torch.nn.functional as F

class Action:
    def __init__(self):
        self.ADD_EDGE = 0
        self.ADD_NODE = 1
    
    def ActionNormlize(self,code):
        return code% (self.ADD_NODE+1)

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias = True, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=bias),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=bias),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x, bn_train=False):
        x = self.ops(x)
        return x

class MaxPool2d(nn.Module):
    def __init__(self, kernelSize, stride, padding):
        super(MaxPool2d, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        padding = 0 if kernelSize in [2] else padding
        self.op = nn.MaxPool2d(kernel_size=kernelSize, stride=stride, padding=padding)

    def forward(self, x):
        if self.kernelSize in [2]:
            x = F.pad(x, self.padding)
        return self.op(x)

class AvgPool2d(nn.Module):
    def __init__(self, kernelSize, stride, padding):
        super(AvgPool2d, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        padding = 0 if kernelSize in [2] else padding
        self.op = nn.AvgPool2d(kernel_size=kernelSize, stride=stride, padding=padding)

    def forward(self, x):
        if self.kernelSize in [2]:
            x = F.pad(x, self.padding)
        return self.op(x)

class Operations:
    def __init__(self):
        self.ops = {
            # 0  : Identity,
            0  : Conv, # 1x1
            1  : Conv, # 3x3 # make sure that distribution of CONV and POOL is 1:1.
            # 3  : Conv, # 1x3 + 3x1
            # 4  : Conv, # 1x7 + 7x1
            2  :  MaxPool2d, # 2x2
            # 6  :  nn.MaxPool2d, # 3x3
            # 7  :  nn.MaxPool2d, # 5x5
            3  :  AvgPool2d, # 2x2
            # 9  :  nn.AvgPool2d, # 3x3
            # 10 :  nn.AvgPool2d, # 5x5
                   }
        self.CONV_kernel_padding = {
            # 0  : [1,0],
            0  : [3,1],
            1  : [5,2],
            2  : [(1,3),((0,1),(1,0))],
            3  : [(1,7),((0,3),(3,0))]
        }
        self.POOL_kernel_padding = {
            0  : [2, [0,1,0,1]],
            1  : [3,1],
            2  : [5,2]
        }
    def getOps(self, opID, channels, kernelSizeID, strideID, bias):
        opID = opID % len(self.ops)
        bias = True if bias%2==0 else False
        # strideID = int((strideID % 2) + 1)
        if opID in [0, 1]:
            kernelSizeID = kernelSizeID % len(self.CONV_kernel_padding)
            kernelSize, padding = self.CONV_kernel_padding[kernelSizeID]
            return self.ops[opID](channels, channels, kernelSize, strideID, padding ,bias)
        else:
            kernelSizeID = kernelSizeID % len(self.POOL_kernel_padding)
            kernelSize, padding = self.POOL_kernel_padding[kernelSizeID]
            return self.ops[opID](kernelSize=kernelSize, stride=strideID, padding=padding)
        
if __name__ == "__main__":
    a = Action()
    b = Operations()
    c = b.getOps(0, 32, 1, 77)
    print(c)
    print('Hello')