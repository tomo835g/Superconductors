import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Periodic_shift_Conv2D(nn.Module):
    """
    If padding=0, and stride=1, then the output size is the same as input size for periodic direction.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True, period_direction=(0, 1), padding_mode='zeros', device=False):
        super(Periodic_shift_Conv2D, self).__init__()
        self.kernel_hight = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.device = device
        self.period_direction = period_direction
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        # input is [batch, channels, height,width]
        batch_dim, channel_dim, height_dim, width_dim = input.size()

        extended_input = torch.zeros(
            batch_dim, channel_dim, height_dim+self.period_direction[0]*(self.kernel_hight-1)+1, width_dim + self.period_direction[1]*(self.kernel_width-1))
        extended_input[:, :,
                       : height_dim, : width_dim] = input
        if self.period_direction[1]:
            extended_input[:, :, 1:, width_dim:] = input[:,
                                                         :, :, : self.kernel_width - 1]
        '''
        dimension of periodic table: batch, channel, height, width
        '''
        # not yet make it what follows.
        # if self.period_direction[0]:
        #     extended_input[:,:,height_dim:,:]=

        # this does not work
        # extended_input = nn.Parameter(extended_input, requires_grad=False)
        # I am skeptical if the following will work in multi GPU
        if self.device:
            extended_input = extended_input.to(self.device)

        # print(extended_input.type())

        output = self.conv(extended_input)
        return output
