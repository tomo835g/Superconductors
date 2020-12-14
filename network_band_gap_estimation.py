import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from periodic_shift_conv2D import PeriodicShiftConv2D


class BlockPeriodicShiftConv2D1(nn.Module):  # 2-layers
    """"
    input size and the output size are designed to be the same
    """

    def __init__(self, input_channels, inside_channels, output_channels, stride=1, padding=(1, 0), kernel_size=[3, 3], device=0):
        super(BlockPeriodicShiftConv2D1, self).__init__()
        self.input_channels = input_channels
        self.inside_channels = inside_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.conv1 = PeriodicShiftConv2D(
            input_channels, inside_channels, kernel_size, stride, padding, device=device)
        self.bn1 = nn.BatchNorm2d(inside_channels)

        self.relu = nn.ReLU()

        self.conv2 = PeriodicShiftConv2D(
            inside_channels, output_channels, kernel_size, stride, padding, device=device)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # self.conv3 =nn.Conv2d(inside_channels,output_channels,kernel_size,stride,padding)
        # self.bn3 =nn.BatchNorm2d(output_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(residual.size())
        # print(out.size())
        out += residual

        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        return out


class ModelRPTWithPeriodicity(nn.Module):
    """
    Here is the model
    this is for band gap estimation. The left- and right-most column is glued up
    """

    def __init__(self, input_channels=4, log=False, binary=False, blocks=20, device=0):
        super(ModelRPTWithPeriodicity, self).__init__()
        # in_channel, out_channel, kernel_size
        self.conv0 = PeriodicShiftConv2D(
            in_channels=input_channels, out_channels=10, device=device)

        # self.bn0 = nn.BatchNorm2d(10)
        self.log = log
        self.binary = binary
        #self.layer_0_1 = BasicBlock_1(input_channels=10, inside_channels=10, output_channels=10)

        #self.block_layer_1 = self._make_layers(channels=4, blocks=blocks)
        self.block_layer_1 = self._make_layers(
            channels=10, blocks=blocks, device=device)

        self.conv1 = nn.Conv2d(10, 20, kernel_size=(
            2, 4), stride=(1, 2), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(20)
        self.layer_1 = BasicPeriodicShiftConv2D1(
            input_channels=20, inside_channels=20, output_channels=20, device=device)

        self.conv2 = nn.Conv2d(20, 30, kernel_size=(2, 4))
        self.bn2 = nn.BatchNorm2d(30)
        self.layer_2 = BasicPeriodicShiftConv2D1(
            input_channels=30, inside_channels=30, output_channels=30, device=device)

        self.conv3 = nn.Conv2d(30, 40, kernel_size=(2, 4))
        self.bn3 = nn.BatchNorm2d(40)
        self.layer_3 = BasicPeriodicShiftConv2D1(
            input_channels=40, inside_channels=40, output_channels=40, device=device)

        self.conv4 = nn.Conv2d(40, 50, kernel_size=(2, 4))
        self.bn4 = nn.BatchNorm2d(50)
        self.layer_4 = BasicPeriodicShiftConv2D1(
            input_channels=50, inside_channels=50, output_channels=50, device=device)

        self.conv5 = nn.Conv2d(50, 60, kernel_size=(2, 4))
        self.bn5 = nn.BatchNorm2d(60)
        self.layer_5 = BasicPeriodicShiftConv2D1(
            input_channels=60, inside_channels=60, output_channels=60, device=device)

        self.conv6 = nn.Conv2d(60, 70, kernel_size=(2, 4))
        self.bn6 = nn.BatchNorm2d(70)
        self.layer_6 = BasicPeriodicShiftConv2D1(
            input_channels=70, inside_channels=70, output_channels=70, device=device)

        # self.conv7 = nn.Conv2d(30, 34, kernel_size=(2, 1))
        # self.bn7 = nn.BatchNorm2d(34)

        self.fc1 = nn.Linear(210, 100)
        # self.fc2 = nn.Linear(100, 1)
        # changed from above to the following
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def _make_layers(self, channels, blocks, device=0):
        layers = []
        for _ in range(blocks):
            layers.append(BasicPeriodicShiftConv2D1(input_channels=channels,
                                                        inside_channels=channels, output_channels=channels, device=device))
        return nn.Sequential(*layers)

    def __L_out1d(self, L_in, kernel_size, stride=1, padding=0, dilation=1):
        temp = L_in + 2 * padding - dilation * kernel_size
        return np.floor(temp / stride + 1)

    def forward(self, x):
        # s = x[:, 0, :, :]  # 2+1
        # p = x[:, 1, :, :]  # 6
        # d = x[:, 2, :, :]  # 9+1
        # f = x[:, 3, :, :]  # 15
        #x = s+p+d+f
        # print('input shape;', x.shape)
        # x = F.relu(self.bn0(self.conv0(x)))
        x = self.conv0(x)
        x = self.block_layer_1(x)

        # x=self.layer_0_1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer_1(x)
        # sel0f.__print(x, do)
        x = F.relu(self.bn2((self.conv2(x))))
        x = self.layer_2(x)

        # self.__print(x, do)
        x = F.relu(self.bn3((self.conv3(x))))
        x = self.layer_3(x)

        x = F.relu(self.bn4((self.conv4(x))))
        x = self.layer_4(x)

        x = F.relu(self.bn5((self.conv5(x))))
        x = self.layer_5(x)

        x = F.relu(self.bn6((self.conv6(x))))
        x = self.layer_6(x)

        #x = F.relu(self.bn7((self.conv7(x))))

        #self.__print(x, do)
        x = x.view(-1, 210)
        # self.__print(x, do)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.log or self.binary:
            x = self.fc3(x)
        else:
            x = F.relu(self.fc3(x))
        # exit()
        return x
