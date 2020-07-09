import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m1 = []
        for i in range(2):
            m1.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m1.append(nn.BatchNorm2d(n_feats))
            if i==0: m1.append(act)

        self.body1 = nn.Sequential(*m1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body1(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsampler_2(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            # m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            # if bn: m.append(nn.BatchNorm2d(n_feats))
            #
            # if act == 'relu':
            #     m.append(nn.ReLU(True))
            # elif act == 'prelu':
            #     m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler_2, self).__init__(*m)
class Upsampler_3(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if scale == 3:
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler_3, self).__init__(*m)

class Upsampler_4(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if scale == 4:
            m.append(nn.PixelShuffle(4))
        else:
            raise NotImplementedError

        super(Upsampler_4, self).__init__(*m)

class Upsampler_8(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if scale == 8:
            m.append(nn.PixelShuffle(8))
        else:
            raise NotImplementedError

        super(Upsampler_8, self).__init__(*m)

class firstLayer_2(nn.Module):
    def __init__(self, conv, n_feats, act=False):

        super(firstLayer_2, self).__init__()
        m = [
            conv(n_feats, n_feats, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        self.body = nn.Sequential(*m)
    def forward(self, x):
        output = [None] * 2
        output[0] = x
        output[1] =  self.body(output[0])
        return output


class MultiBlock_2(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(MultiBlock_2, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]
        up = [Upsampler_2(conv, 2, n_feats, act=False)]

        fs = [conv(2*64, 64, 1)]

        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)
        self.fs = nn.Sequential(*fs)
    def forward(self, x):
        output = [None] * 2
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = self.fs(torch.cat([x[1], self.up(output[0])], 1))
        output[1] += x[1]
        return output

class lastMultiBlock_2(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastMultiBlock_2, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]
        up = [Upsampler(conv, 2, n_feats, act=False)]
        fs = [conv(320, 256, 1)]
        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)
        self.fs = nn.Sequential(*fs)
    def forward(self, x):
        output = [None] * 2
        # cur_input = x[0]
        output[0] = self.res (x[0])
        output[1] = self.fs(torch.cat([x[1], self.up(output[0])], 1))
        return output

class lastLayer_2(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastLayer_2, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ),
            conv(n_feats, n_feats, kernel_size)
        ]
        self.res = nn.Sequential(*res)

    def forward(self, x):
        output = [None] * 2
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = x[1]
        return output

class reconstruction_2(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=False):

        super(reconstruction_2, self).__init__()
        up = [
            Upsampler(conv, 2, n_feats, act=act)
        ]

        tail = [
            conv(2*n_feats, n_feats, 1),
            conv(n_feats, 3, kernel_size)

        ]

        self.up = nn.Sequential(*up)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        # cur_input = x[0]
        output = self.tail(torch.cat([self.up(x[0]),x[1]], 1))
        return output

class firstLayer_3(nn.Module):
    def __init__(self, conv, n_feats, act=False):

        super(firstLayer_3, self).__init__()
        m = [
            conv(n_feats, n_feats, 3, bias=True),
            Upsampler_3(conv, 3, n_feats, act=act)
        ]
        self.body = nn.Sequential(*m)
    def forward(self, x):
        output = [None] * 2
        output[0] = x
        output[1] =  self.body(output[0])
        return output


class MultiBlock_3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(MultiBlock_3, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]
        up = [Upsampler_3(conv, 3, n_feats, act=False)]

        fs = [conv(2*28, 28, 1)]

        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)
        self.fs = nn.Sequential(*fs)
    def forward(self, x):
        output = [None] * 2
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = self.fs(torch.cat([x[1], self.up(output[0])], 1))
        output[1] += x[1]
        return output

class lastMultiBlock_3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastMultiBlock_3, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]
        up = [Upsampler(conv, 3, n_feats, act=False)]
        fs = [conv(280, n_feats, 1)]
        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)
        self.fs = nn.Sequential(*fs)
    def forward(self, x):
        output = [None] * 2
        output[0] = self.res (x[0])
        output[1] = self.fs(torch.cat([x[1], self.up(output[0])], 1))
        return output

class lastLayer_3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastLayer_3, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ),
            conv(n_feats, n_feats, kernel_size)
        ]
        self.res = nn.Sequential(*res)

    def forward(self, x):
        output = [None] * 2
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = x[1]
        return output

class reconstruction_3(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=False):

        super(reconstruction_3, self).__init__()
        up = [
            Upsampler(conv, 3, n_feats, act=act)
        ]

        tail = [
            conv(2*n_feats, n_feats, 1),
            conv(n_feats, 3, kernel_size)

        ]

        self.up = nn.Sequential(*up)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        # cur_input = x[0]
        output = self.tail(torch.cat([self.up(x[0]),x[1]], 1))
        return output

# Definition of the model of x4
class firstLayer_4(nn.Module):
    def __init__(self, conv, n_feats, act=False):

        super(firstLayer_4, self).__init__()
        m_up2_1 = [
            conv(n_feats, n_feats, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        m_up2_2 = [
            conv(64, 64, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        self.up2_1 = nn.Sequential(*m_up2_1)
        self.up2_2 = nn.Sequential(*m_up2_2)
    def forward(self, x):
        output = [None] * 3
        output[0] = x
        output[1] = self.up2_1(output[0])
        output[2] = self.up2_2(output[1])

        return output

class MultiBlock_4(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(MultiBlock_4, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]
        up2 = [Upsampler_2(conv, 2, n_feats, act=False)]
        up4 = [Upsampler_4(conv, 4, n_feats, act=False)]

        fs_1 = [conv(2*64, 64, 1)]
        fs_2 = [conv(2*16, 16, 1)]
        self.res = nn.Sequential(*res)
        self.up2 = nn.Sequential(*up2)
        self.up4 = nn.Sequential(*up4)
        self.fs_1 = nn.Sequential(*fs_1)
        self.fs_2 = nn.Sequential(*fs_2)

    def forward(self, x):
        output = [None] * 3
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = self.fs_1(torch.cat([x[1], self.up2(output[0])], 1)) + x[1]
        # cur_input = output[1]
        output[2] = self.fs_2(torch.cat([x[2], self.up2(output[1])], 1)) + x[2]
        output[2] += self.up4(x[0])
        return output

class lastMultiBlock_4(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastMultiBlock_4, self).__init__()
        res = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            )
        ]

        up_0 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        up_1 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        fs_1 = [conv(320, n_feats, 1)]
        fs_2 = [conv(272, n_feats, 1)]

        self.res = nn.Sequential(*res)
        self.up2_0 = nn.Sequential(*up_0)
        self.up2_1 = nn.Sequential(*up_1)
        self.fs_1 = nn.Sequential(*fs_1)
        self.fs_2 = nn.Sequential(*fs_2)

    def forward(self, x):
        output = [None] * 3
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = self.fs_1(torch.cat([x[1], self.up2_0(output[0])], 1))
        # cur_input = output[1]
        output[2] = self.fs_2(torch.cat([x[2], self.up2_1(output[1])], 1))

        return output

class lastLayer_4(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1):

        super(lastLayer_4, self).__init__()

        res1 = [
            conv(n_feats, n_feats, kernel_size, bias=True),
            nn.ReLU(inplace=True),
            conv(n_feats, n_feats, kernel_size, bias=True)
        ]
        res1.append(conv(n_feats, n_feats, kernel_size))

        self.res = nn.Sequential(*res1)

    def forward(self, x):
        output = [None] * 3
        cur_input = x[0]
        output[0] = self.res(cur_input)

        output[1] = x[1]
        output[2] = x[2]
        return output

class reconstruction_4(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=False):

        super(reconstruction_4, self).__init__()
        up2_0 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        fs = [
            conv(2 * n_feats, n_feats, 1)
        ]
        up2_1 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        tail = [
            conv(2*n_feats, n_feats, 1),
            conv(n_feats, 3, kernel_size)
        ]

        self.up2_0 = nn.Sequential(*up2_0)
        self.fs = nn.Sequential(*fs)
        self.up2_1 = nn.Sequential(*up2_1)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        # cur_input = x[0]
        output = self.fs(torch.cat([self.up2_0(x[0]),x[1]], 1))
        output = self.tail(torch.cat([self.up2_1(output),x[2]],1))
        return output


# # Definition of the model of x8
class firstLayer_8(nn.Module):
    def __init__(self, conv, n_feats, act=False):

        super(firstLayer_8, self).__init__()
        m_up2_1 = [
            conv(n_feats, n_feats, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        m_up2_2 = [
            conv(64, 64, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        m_up2_3 = [
            conv(16, 16, 3, bias=True),
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        self.up2_1 = nn.Sequential(*m_up2_1)
        self.up2_2 = nn.Sequential(*m_up2_2)
        self.up2_3 = nn.Sequential(*m_up2_3)

    def forward(self, x):
        output = [None] * 4
        output[0] = x
        output[1] = self.up2_1(output[0])
        output[2] = self.up2_2(output[1])
        output[3] = self.up2_3(output[2])
        return output

class MultiBlock_8(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(MultiBlock_8, self).__init__()
        m_res = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        ]
        m_up2 = [
            Upsampler_2(conv, 2, n_feats, act=act)
        ]
        m_up4 = [
            Upsampler_4(conv, 4, n_feats, act=act)
        ]
        m_up8 = [
            Upsampler_8(conv, 8, n_feats, act=act)
        ]
        m_fs_1 = [conv(2 * 64, 64, 1)]
        m_fs_2 = [conv(2 * 16, 16, 1)]
        m_fs_3 = [conv(2 * 4, 4, 1)]
        self.res = nn.Sequential(*m_res)
        self.up2 = nn.Sequential(*m_up2)
        self.fs_1 = nn.Sequential(*m_fs_1)
        self.fs_2 = nn.Sequential(*m_fs_2)
        self.fs_3 = nn.Sequential(*m_fs_3)
        self.up4 = nn.Sequential(*m_up4)
        self.up8 = nn.Sequential(*m_up8)
    def forward(self, x):
        output = [None] * 4
        cur_input = x[0]
        output[0] = self.res(cur_input)
        output[1] = self.fs_1(torch.cat([x[1], self.up2(output[0])], 1)) + x[1]
        # cur_input = output[1]
        output[2] = self.fs_2(torch.cat([x[2], self.up2(output[1])], 1)) + x[2]
        output[2] += self.up4(output[0])
        # cur_input = output[2]
        output[3] = self.fs_3(torch.cat([x[3], self.up2(output[2])], 1)) + x[3]
        output[3] += self.up8(output[0])
        return output

class lastMultiBlock_8(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=nn.ReLU(True),res_scale=1):

        super(lastMultiBlock_8, self).__init__()
        res = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        ]
        up_0 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        up_1 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        up_2 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        fs_1 = [conv(320, n_feats, 1)]
        fs_2 = [conv(272, n_feats, 1)]
        fs_3 = [conv(260, n_feats, 1)]
        self.res = nn.Sequential(*res)
        self.up2_0 = nn.Sequential(*up_0)
        self.fs_1 = nn.Sequential(*fs_1)
        self.up2_1 = nn.Sequential(*up_1)
        self.fs_2 = nn.Sequential(*fs_2)
        self.up2_2 = nn.Sequential(*up_2)
        self.fs_3 = nn.Sequential(*fs_3)

    def forward(self, x):
        output = [None] * 4
        cur_input = x[0]
        output[0] = self.res(cur_input)
        output[1] = self.fs_1(torch.cat([x[1], self.up2_0(output[0])], 1))
        output[2] = self.fs_2(torch.cat([x[2], self.up2_1(output[1])], 1))
        output[3] = self.fs_3(torch.cat([x[3], self.up2_2(output[2])], 1))
        return output

class lastLayer_8(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=False, res_scale = 1):

        super(lastLayer_8, self).__init__()
        res1 = [
            conv(n_feats, n_feats, kernel_size, bias=True),
            nn.ReLU(inplace=True),
            conv(n_feats, n_feats, kernel_size, bias=True)
        ]
        res1.append(conv(n_feats, n_feats, kernel_size))

        self.res = nn.Sequential(*res1)

    def forward(self, x):
        output = [None] * 4
        # cur_input = x[0]
        output[0] = self.res(x[0])
        output[1] = x[1]
        output[2] = x[2]
        output[3] = x[3]
        return output

class reconstruction_8(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, act=False):

        super(reconstruction_8, self).__init__()

        up2_0 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        up2_1 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        up2_2 = [
            Upsampler(conv, 2, n_feats, act=act)
        ]
        fs_0 = [
            conv(2 * n_feats, n_feats, 1)
        ]
        fs_1 = [
            conv(2 * n_feats, n_feats, 1)
        ]
        tail = [
            conv(2*n_feats, n_feats, 1),
            conv(n_feats, 3, kernel_size)

        ]

        self.up2_0 = nn.Sequential(*up2_0)
        self.up2_1 = nn.Sequential(*up2_1)
        self.up2_2 = nn.Sequential(*up2_2)
        self.fs_0 = nn.Sequential(*fs_0)
        self.fs_1 = nn.Sequential(*fs_1)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        # cur_input = x[0]
        output = self.fs_0(torch.cat([self.up2_0(x[0]),x[1]], 1))
        output = self.fs_1(torch.cat([self.up2_1(output), x[2]],1))
        output = self.tail(torch.cat([self.up2_2(output), x[3]], 1))
        return output

