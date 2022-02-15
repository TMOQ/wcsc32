import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dlshogi.common import *

__all__ = ['ghost_net']

k = 192
fcl = 256 # fully connected layers

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            #depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, 1, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                #nn.Conv2d(inp, oup, 1, 1, 1, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)





class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        k = 192
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        #self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        #self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        #self.norm1 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm2 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm3 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm4 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm5 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm6 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm7 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm8 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm9 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        #self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)

        #super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        #self.cfgs = cfgs

        # building first layer
        #output_channel = _make_divisible(16 * width_mult, 4)
        output_channel = k
        input_channel = output_channel
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        layers = [nn.Sequential(
            #nn.Conv2d(input_channel, output_channel, 3, 2, 1, bias=False),
            nn.Conv2d(input_channel, output_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        #input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        #for k, exp_size, c, use_se, s in self.cfgs:
        for rep_ in range(10):
            output_channel = k #_make_divisible(c * width_mult, 4)
            hidden_channel = k #_make_divisible(exp_size * width_mult, 4)
            #layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            layers.append(block(input_channel, hidden_channel, output_channel, 3, 1, 0))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

    def forward(self, x1, x2):
        #print("x1: ", x1.shape)
        #print("x2: ", x2.shape)
        u1_1_1 = self.l1_1_1(x1)
        #print("u1_1_1: ", u1_1_1.shape)
        u1_1_2 = self.l1_1_2(x1)
        #print("u1_1_2: ", u1_1_2.shape)
        u1_2 = self.l1_2(x2)
        #print("u1_2: ", u1_2.shape)
        u1 = u1_1_1 + u1_1_2 + u1_2
        #print("u1: ", u1.shape)
        # Ghost block
        u21 = self.features(u1)
        #print("u21: ", u21.shape)

        # Residual block
        #h1 = F.relu(self.norm1(u1))
        #h2 = F.relu(self.norm2(self.l2(h1)))
        #u3 = self.l3(h2) + u1
        # Residual block
        #h3 = F.relu(self.norm3(u3))
        #h4 = F.relu(self.norm4(self.l4(h3)))
        #u5 = self.l5(h4) + u3
        # Residual block
        #h5 = F.relu(self.norm5(u5))
        #h6 = F.relu(self.norm6(self.l6(h5)))
        #u7 = self.l7(h6) + u5
        # Residual block
        #h7 = F.relu(self.norm7(u7))
        #h8 = F.relu(self.norm8(self.l8(h7)))
        #u9 = self.l9(h8) + u7
        # Residual block
        #h9 = F.relu(self.norm9(u9))
        #h10 = F.relu(self.norm10(self.l10(h9)))
        #u11 = self.l11(h10) + u9
        # Residual block
        #h11 = F.relu(self.norm11(u11))
        #h12 = F.relu(self.norm12(self.l12(h11)))
        #u13 = self.l13(h12) + u11
        # Residual block
        #h13 = F.relu(self.norm13(u13))
        #h14 = F.relu(self.norm14(self.l14(h13)))
        #u15 = self.l15(h14) + u13
        # Residual block
        #h15 = F.relu(self.norm15(u15))
        #h16 = F.relu(self.norm16(self.l16(h15)))
        #u17 = self.l17(h16) + u15
        # Residual block
        #h17 = F.relu(self.norm17(u17))
        #h18 = F.relu(self.norm18(self.l18(h17)))
        #u19 = self.l19(h18) + u17
        # Residual block
        #h19 = F.relu(self.norm19(u19))
        #h20 = F.relu(self.norm20(self.l20(h19)))
        #u21 = self.l21(h20) + u19

        h21 = F.relu(self.norm21(u21))
        # policy network
        h22 = self.l22(h21)
        h22_1 = self.l22_2(torch.flatten(h22, 1))
        # value network
        h22_v = F.relu(self.norm22_v(self.l22_v(h21)))
        h23_v = F.relu(self.l23_v(torch.flatten(h22_v, 1)))
        return h22_1, self.l24_v(h23_v)
