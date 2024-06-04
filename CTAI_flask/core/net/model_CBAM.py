import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

model_eff = EfficientNet.from_pretrained('efficientnet-b0')

net1 = nn.Sequential(
    *list(model_eff.children())[:2], *((list(model_eff.children())[2:3])[0])[:1]  # 16,112,112
)

net2 = nn.Sequential(
    *((list(model_eff.children())[2:3])[0])[1:3]  # 24,56,56
)
net3 = nn.Sequential(
    *((list(model_eff.children())[2:3])[0])[3:5]  # 40,28,28
)
net4 = nn.Sequential(
    *((list(model_eff.children())[2:3])[0])[5:11]  # 112,14,14
)
net5 = nn.Sequential(
    *((list(model_eff.children())[2:3])[0])[11:16], *(list(model_eff.children())[3:5])  # 1280,7,7
)


# 通道注意力
class channel_attention(nn.Module):
    def __init__(self, channel, ration=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # torch.Size([2, 64, 5, 5])
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])  # torch.Size([2, 64])
        max_pool = self.max_pool(x).view([b, c])  # torch.Size([2, 64])

        avg_fc = self.fc(avg_pool)  # torch.Size([2, 64])
        max_fc = self.fc(max_pool)  # torch.Size([2, 64])

        out = self.sigmoid(max_fc + avg_fc).view([b, c, 1, 1])  ##torch.Size([2, 64, 1, 1])
        return x * out


# 空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 通道的最大池化
        max_pool = torch.max(x, dim=1, keepdim=True).values  # torch.Size([2, 1, 5, 5])
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # torch.Size([2, 1, 5, 5])
        pool_out = torch.cat([max_pool, avg_pool], dim=1)  # torch.Size([2, 2, 5, 5])
        conv = self.conv(pool_out)  # torch.Size([2, 1, 5, 5])
        out = self.sigmoid(conv)

        return out * x


# 将通道注意力和空间注意力进行融合
class CBAM(nn.Module):
    def __init__(self, channel, ration=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)  # torch.Size([2, 64, 5, 5])
        out = self.spatial_attention(out)  # torch.Size([2, 64, 5, 5])

        return out


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, inputs, catputs):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        up_out = self.upsample(inputs)
        cat_out = torch.cat((up_out, catputs), dim=1)
        conv_out = self.Conv_BN_ReLU_2(cat_out)
        return conv_out


class EffB0_UNet(nn.Module):
    def __init__(self):
        super(EffB0_UNet, self).__init__()
        out_channels = [16, 24, 40, 112, 1280]
        # 下采样
        self.d1 = nn.Sequential(net1, CBAM(out_channels[0]))  # 112, 112, 16
        self.d2 = nn.Sequential(net2, CBAM(out_channels[1]))  # 56, 56, 24
        self.d3 = nn.Sequential(net3, CBAM(out_channels[2]))  # 28, 28, 40
        self.d4 = nn.Sequential(net4, CBAM(out_channels[3]))  # 14, 14, 112
        self.d5 = nn.Sequential(net5, CBAM(out_channels[4], ))  # 7, 7, 1280

        # 上采样
        self.u1 = UpSampleLayer(out_channels[4], out_channels[3])  # 1280 -> 112
        self.u2 = UpSampleLayer(out_channels[3], out_channels[2])  # 112 -> 40
        self.u3 = UpSampleLayer(out_channels[2], out_channels[1])  # 40 -> 24
        self.u4 = UpSampleLayer(out_channels[1], out_channels[0])  # 24 -> 16
        # 输出
        self.o = nn.Sequential(
            nn.ConvTranspose2d(out_channels[0], 1, 2, stride=2, padding=0),
            nn.Sigmoid()
            # BCELoss
        )

    def forward(self, x):
        out1 = self.d1(x)  # 112 112 16
        out2 = self.d2(out1)  # 56 56 24
        out3 = self.d3(out2)  # 28 28 32
        out4 = self.d4(out3)  # 14 14 96
        out5 = self.d5(out4)  # 7 7 1280

        up_out1 = self.u1(out5, out4)  # 14 14 96
        up_out2 = self.u2(up_out1, out3)  # 28 28 32
        up_out3 = self.u3(up_out2, out2)  # 56 56 24
        up_out4 = self.u4(up_out3, out1)  # 112 112 16
        out = self.o(up_out4)
        return out
