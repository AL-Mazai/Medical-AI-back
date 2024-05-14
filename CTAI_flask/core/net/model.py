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


# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Applying projection if needed
        if self.projection is not None:
            residual = self.projection(residual)

        out += residual
        out = self.relu(out)
        return out


class EffB0_UNet(nn.Module):
    def __init__(self):
        super(EffB0_UNet, self).__init__()
        out_channels = [16, 24, 40, 112, 1280]
        # 下采样
        self.d1 = net1  # 112 112 16
        self.d2 = net2  # 56 56 24
        self.d3 = net3  # 28 28 32
        self.d4 = net4  # 14 14 96
        self.d5 = net5  # 7 7 1280

        # 上采样
        self.u1 = UpSampleLayer(out_channels[4], out_channels[3])  # 1280-96*2-96
        self.u2 = UpSampleLayer(out_channels[3], out_channels[2])  # 96-32*2-32
        self.u3 = UpSampleLayer(out_channels[2], out_channels[1])  # 32-24*2-24
        self.u4 = UpSampleLayer(out_channels[1], out_channels[0])  # 24-16*2-16
        # 输出
        self.o = nn.Sequential(
            nn.ConvTranspose2d(out_channels[0], 1, 2, stride=2, padding=0),
            nn.Sigmoid()
            # BCELoss
        )
        self.attention = SelfAttention(1280)

    def forward(self, x):
        out1 = self.d1(x)  # 112 112 16
        out2 = self.d2(out1)  # 56 56 24
        out3 = self.d3(out2)  # 28 28 32
        out4 = self.d4(out3)  # 14 14 96
        out5 = self.d5(out4)  # 7 7 1280

        # 注意力
        out5 = self.attention(out5)

        up_out1 = self.u1(out5, out4)  # 14 14 96
        up_out2 = self.u2(up_out1, out3)  # 28 28 32
        up_out3 = self.u3(up_out2, out2)  # 56 56 24
        up_out4 = self.u4(up_out3, out1)  # 112 112 16
        out = self.o(up_out4)
        return out
