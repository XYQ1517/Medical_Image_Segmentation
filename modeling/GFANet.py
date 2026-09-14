import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


__C = AttrDict()
# Consumers can get config by:

cfg = __C
__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d
__C.MODEL.BIGMEMORY = False


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalizationLayer = layer(in_channels)
    return normalizationLayer


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))

        x = self.relu(x_cat + self.conv_res(x))

        return x


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        # print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


# -

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=True, weight_path=''):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                      stride=2, bias=False))

        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000)))

    def forward(self, x):
        out_branch = []

        for i in range(len(self.base) - 1):
            x = self.base[i](x)
            # out_branch.append(x)
            if i == 1 or i == 4 or i == 9 or i == 12 or i == 15:
                out_branch.append(x)

        return out_branch


def hardnet(arch=68, pretrained=True, **kwargs):
    if arch == 68:
        # print("68 LOADED")
        model = HarDNet(arch=68)
        if pretrained:
            checkpoint = torch.load('modeling/hardnet68.pth',
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint, strict=False)
            # print("68 LOADED READY")

    return model


# --------------- MultiSpectralAttentionLayer -------------------

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            Norm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            Norm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                  bias=False),
        norm_layer(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        nn.Sigmoid()
    )


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class DenseBlock(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(DenseBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                           dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature


class GFD(nn.Module):
    def __init__(self, num_classes=2, norm_layer=nn.BatchNorm2d):
        super(GFD, self).__init__()

        self.d_in1 = conv_bn_relu(36, 36, 1, norm_layer=norm_layer)
        self.d_in2 = conv_bn_relu(36, 36, 1, norm_layer=norm_layer)
        self.d_in3 = conv_bn_relu(36, 36, 1, norm_layer=norm_layer)

        self.gate1 = conv_sigmoid(36, 36)
        self.gate2 = conv_sigmoid(36, 36)
        self.gate3 = conv_sigmoid(36, 36)

        in_channel = 36
        self.dense_3 = DenseBlock(in_channel, 36, 36, 3, drop_out=0, norm_layer=norm_layer)
        self.dense_6 = DenseBlock(in_channel + 36, 36, 36, 6, drop_out=0, norm_layer=norm_layer)
        self.dense_9 = DenseBlock(in_channel + 36 * 2, 36, 36, 9, drop_out=0, norm_layer=norm_layer)

        self.cls = nn.Sequential(
            nn.Conv2d(144, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

    def forward(self, m2, m5, aspp):
        m2_size = m2.size()[2:]
        m5_size = m5.size()[2:]
        aspp_size = aspp.size()[2:]

        g_m2 = self.gate1(m2)
        g_m5 = self.gate2(m5)
        g_aspp = self.gate3(aspp)

        m2 = self.d_in1(m2)
        m5 = self.d_in2(m5)
        aspp = self.d_in3(aspp)

        m2 = m2 + g_m2 * m2 + (1 - g_m2) * (Upsample(g_m5 * m5, size=m2_size) + Upsample(g_aspp * aspp, size=m2_size))
        m5 = m5 + g_m5 * m5 + (1 - g_m5) * (
                Upsample(g_m2 * m2, size=m5_size) + Upsample(g_aspp * aspp, size=m5_size))
        aspp_f = aspp + aspp * g_aspp + (1 - g_aspp) * (
                Upsample(g_m5 * m5, size=aspp_size) + Upsample(g_m2 * m2, size=aspp_size))

        aspp_f = Upsample(aspp_f, size=m2_size)
        aspp = Upsample(aspp, size=m2_size)
        m5 = Upsample(m5, size=m2_size)

        out = aspp_f
        aspp_f = self.dense_3(out)
        out = torch.cat([aspp_f, m5], dim=1)
        m5 = self.dense_6(out)
        out = torch.cat([aspp_f, m5, m2], dim=1)
        m2 = self.dense_9(out)

        f = torch.cat([aspp_f, aspp, m5, m2], dim=1)

        out = self.cls(f)

        return out


class GFANet(nn.Module):
    def __init__(self, num_classes=1, channel=32):
        super(GFANet, self).__init__()

        self.hardnet = hardnet(arch=68)

        # ------ CEM ------
        self.rfb2_1 = RFB_modified(320, channel)
        self.spp2_1 = SPPblock(channel)

        self.rfb3_1 = RFB_modified(640, channel)
        self.spp3_1 = SPPblock(channel)

        self.rfb4_1 = RFB_modified(1024, channel)
        self.spp4_1 = SPPblock(channel)

        # ------ GFD ------
        self.gff_head = GFD(num_classes=2, norm_layer=Norm2d)
        initialize_weights(self.gff_head)

        # ------ chanel reverse attention branch 4 ------
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 2, kernel_size=1)
        self.att4 = MultiSpectralAttentionLayer(256, 7, 10, reduction=16, freq_sel_method='top16')

        # ------ chanel reverse attention branch 3 ------
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 2, kernel_size=3, padding=1)
        self.att3 = MultiSpectralAttentionLayer(64, 14, 20, reduction=16, freq_sel_method='top16')

        # ------ chanel reverse attention branch 2 ------
        self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 2, kernel_size=3, padding=1)
        self.att2 = MultiSpectralAttentionLayer(64, 28, 40, reduction=16, freq_sel_method='top16')

        # ------ GCF ------
        self.gate1 = GatedSpatialConv2d(256, 256)
        self.gate2 = GatedSpatialConv2d(64, 64)
        self.gate3 = GatedSpatialConv2d(64, 64)
        self.gate4 = GatedSpatialConv2d(128, 128)

        self.final_seg = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final_seg)

    def forward(self, x):

        hardnetout = self.hardnet(x)

        x0 = hardnetout[0]  # (b, 64, 112, 160)
        x1 = hardnetout[1]  # (b, 128, 56, 80)
        x2 = hardnetout[2]  # (b, 320, 28, 40)
        x3 = hardnetout[3]  # (b, 640, 14, 20)
        x4 = hardnetout[4]  # (b, 1024, 7, 10)

        x2_rfb = self.rfb2_1(x2)
        x2_spp = self.spp2_1(x2_rfb)

        x3_rfb = self.rfb3_1(x3)
        x3_spp = self.spp3_1(x3_rfb)

        x4_rfb = self.rfb4_1(x4)
        x4_spp = self.spp4_1(x4_rfb)

        ra5_feat = self.gff_head(x2_spp, x3_spp, x4_spp)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')

        # ------ chanel reverse attention branch_4 ------
        ra5_feat = ra5_feat[:, 1:, :, :]
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = self.att4(x)

        g1 = self.gate1(x, crop_4)

        x = F.relu(self.ra4_conv2(g1))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)

        lateral_map_4 = F.interpolate(ra4_feat, scale_factor=32, mode='bilinear')

        # ------ chanel reverse attention branch_3 ------
        ra4_feat = ra4_feat[:, 1:, :, :]
        crop_3 = F.interpolate(ra4_feat, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 640, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = self.att3(x)

        g2 = self.gate2(x, crop_3)

        x = F.relu(self.ra3_conv2(g2))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        lateral_map_3 = F.interpolate(ra3_feat, scale_factor=16, mode='bilinear')

        # ------ chanel reverse attention branch_2 ------
        ra3_feat = ra3_feat[:, 1:, :, :]
        crop_2 = F.interpolate(ra3_feat, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 320, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = self.att2(x)

        g3 = self.gate3(x, crop_2)

        x = F.relu(self.ra2_conv2(g3))
        ra3_fuse = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(ra3_fuse)

        lateral_map_2 = F.interpolate(ra2_feat, scale_factor=8, mode='bilinear')

        ra2_feat = ra2_feat[:, 1:, :, :]
        crop_1 = F.interpolate(ra2_feat, scale_factor=2, mode='bilinear')

        g4 = self.gate4(x1, crop_1)

        dec0 = self.final_seg(g4)
        seg_out = F.interpolate(dec0, scale_factor=4, mode='bilinear')

        # return torch.sigmoid(lateral_map_5), torch.sigmoid(lateral_map_4), torch.sigmoid(lateral_map_3), \
        #        torch.sigmoid(lateral_map_2), torch.sigmoid(seg_out)
        return torch.sigmoid(seg_out)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    x = torch.randn((1, 3, 256, 256)).to("cuda:0")
    model = GFANet().to("cuda:0")
    y = model(x)
    print(y.shape)
    MACs, Params = profile(model, inputs=(x,), verbose=False)
    Flops, Params = clever_format([MACs * 2, Params], '%.2f')
    print(f"Flops:{Flops}")
    print(f"Params:{Params}")
