# UNET model (5 layers) with configurable input channels for MAE feature concatenation

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = AttentionGate(in_c, out_c)
        self.c1 = ConvBlock(in_c[0] + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x


class UNETWithExtraChannels(nn.Module):
    """
    与原始 5 层 U-Net 架构等价，但支持自定义输入通道数（用于拼接 MAE 特征）。

    使用方式：当拼接 k 个 MAE 特征通道时，in_channels 传入 1+k。
    其余结构保持不变，以尽量减少对现有训练/评估代码的影响。
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 1024)

        self.b1 = ConvBlock(1024, 2048)

        self.d1 = DecoderBlock([2048, 1024], 1024)
        self.d2 = DecoderBlock([1024, 512], 512)
        self.d3 = DecoderBlock([512, 256], 256)
        self.d4 = DecoderBlock([256, 128], 128)
        self.d5 = DecoderBlock([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)

        b1 = self.b1(p5)

        d1 = self.d1(b1, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)

        output = self.output(d5)
        return output


