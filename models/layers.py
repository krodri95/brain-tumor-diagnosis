import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, att=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        self.att = nn.Identity() if att is None else (SqueezeExcitation(c2, int(c2/4)) if att == 'SE' else CBAM(c2, ratio=4))

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.channels = channels

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels)
            )
        self.pool_types = pool_types
        self.act = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.globalAvgPool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = self.globalMaxPool(x)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.act(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.act(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, ratio=4, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()
        self.ch_att = ChannelAttention(channels, ratio, pool_types)
        self.spatial=spatial
        if spatial:
            self.sp_att = SpatialAttention()

    def forward(self, x):
        x = self.ch_att(x)
        if self.spatial:
            x = self.sp_att(x)
        return x


class FeaturePyramidBlock(nn.Module):
    def __init__(self, channels, att=None):
        super().__init__()
        self.conv0 = Conv(channels[0], 32, 3, att=att)
        self.conv1 = Conv(32+channels[1], 32, 3, att=att)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.avgpool0 = nn.AdaptiveAvgPool2d(8)
        self.avgpool1 = nn.AdaptiveMaxPool2d(8)
        self.convf = Conv(64, 32, 1, att=att)

    def forward(self, x0, x1):
        x0_ = self.conv0(x0)
        x00_ = self.avgpool0(x0_)
        x11_ = self.avgpool1(self.conv1(torch.cat((self.up(x0_),x1), dim=1)))
        x = torch.cat((x00_, x11_), dim=1)
        x = self.convf(x).flatten(1)
        return x
