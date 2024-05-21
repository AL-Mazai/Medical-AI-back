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


class SENet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])
        return x * fc


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
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.se = SENet(out_ch)

    def forward(self, inputs, catputs):
        up_out = self.upsample(inputs)
        cat_out = torch.cat((up_out, catputs), dim=1)
        conv_out = self.Conv_BN_ReLU_2(cat_out)
        se_out = self.se(conv_out)
        return se_out


class EffB0_UNet(nn.Module):
    def __init__(self):
        super(EffB0_UNet, self).__init__()
        out_channels = [16, 24, 40, 112, 1280]
        self.d1 = net1  # 112 112 16
        self.d2 = net2  # 56 56 24
        self.d3 = net3  # 28 28 32
        self.d4 = net4  # 14 14 96
        self.d5 = net5  # 7 7 1280

        self.se1 = SENet(out_channels[0])
        self.se2 = SENet(out_channels[1])
        self.se3 = SENet(out_channels[2])
        self.se4 = SENet(out_channels[3])
        self.se5 = SENet(out_channels[4])

        self.u1 = UpSampleLayer(out_channels[4], out_channels[3])  # 1280-96*2-96
        self.u2 = UpSampleLayer(out_channels[3], out_channels[2])  # 96-32*2-32
        self.u3 = UpSampleLayer(out_channels[2], out_channels[1])  # 32-24*2-24
        self.u4 = UpSampleLayer(out_channels[1], out_channels[0])  # 24-16*2-16

        self.o = nn.Sequential(
            nn.ConvTranspose2d(out_channels[0], 1, 2, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.d1(x)  # 112 112 16
        out1 = self.se1(out1)
        out2 = self.d2(out1)  # 56 56 24
        out2 = self.se2(out2)
        out3 = self.d3(out2)  # 28 28 32
        out3 = self.se3(out3)
        out4 = self.d4(out3)  # 14 14 96
        out4 = self.se4(out4)
        out5 = self.d5(out4)  # 7 7 1280
        out5 = self.se5(out5)

        up_out1 = self.u1(out5, out4)  # 14 14 96
        up_out2 = self.u2(up_out1, out3)  # 28 28 32
        up_out3 = self.u3(up_out2, out2)  # 56 56 24
        up_out4 = self.u4(up_out3, out1)  # 112 112 16
        out = self.o(up_out4)
        return out
