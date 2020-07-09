from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return PRNetx3(args)

class PRNetx3(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(PRNetx3, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats # the channel numbers
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        #self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_head_1 =[
            common.firstLayer_3(conv, n_feats, act=False)
        ]

        # define body module
        m_body=[
            common.MultiBlock_3(
                conv, n_feats, kernel_size, res_scale = args.res_scale
            ) for _ in range (n_resblocks-2)
        ]

        m_body_1 = [
            common.lastMultiBlock_3(
                conv, n_feats, kernel_size, res_scale=args.res_scale
            )
        ]

        m_body_2 = [
            common.lastLayer_3(
                conv, n_feats, kernel_size, res_scale=args.res_scale
            )
        ]

        # define tail module
        m_tail = [
            common.reconstruction_3(conv, n_feats, kernel_size, act=False)

        ]

        self.head = nn.Sequential(*m_head)
        self.head_1 = nn.Sequential(*m_head_1)
        self.body = nn.Sequential(*m_body)
        self.body_1 = nn.Sequential(*m_body_1)
        self.body_2 = nn.Sequential(*m_body_2)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        head = self.head(x)
        head_1 = self.head_1(head)
        res = self.body(head_1)
        res = self.body_1(res)
        res = self.body_2(res)
        res[0] = res[0] + head
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

