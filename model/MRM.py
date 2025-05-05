import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResBottleneck(nn.Module):
    """多分辨率瓶颈模块"""
    def __init__(self, in_ch, out_ch):
        super(MultiResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.conv_pool = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out_pool = self.conv_pool(self.pool(x))
        return out1 + out3 + F.interpolate(out_pool, size=out1.shape[2:], mode='bilinear')


class RecurrentDecoder(nn.Module):
    """循环解码器模块"""
    def __init__(self, in_ch, out_ch):
        super(RecurrentDecoder, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.recurrent = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        for _ in range(3):  # 进行3次循环优化
            x = self.recurrent(F.relu(x))
        return x


class RefUnet(nn.Module):
    """优化后的 RefUnet，使用多分辨率瓶颈层和循环解码器"""
    def __init__(self, in_ch, inc_ch, num_classes):
        super(RefUnet, self).__init__()
        # 编码器部分
        self.conv0 = nn.Conv2d(in_ch, inc_ch, kernel_size=3, padding=1)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(inc_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 多分辨率瓶颈层
        self.bottleneck = MultiResBottleneck(64, 64)

        # 循环解码器部分
        self.decoder = RecurrentDecoder(64, 64)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        # 上采样层，确保输出分辨率与输入一致
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 编码阶段
        residual = x
        x = self.conv0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 多分辨率瓶颈层
        bottleneck_output = self.bottleneck(e4)

        # 循环解码器
        decoded = self.decoder(bottleneck_output)

        # 最终输出层
        output = self.final_conv(decoded)

        # 上采样到输入大小
        output = self.upsample(output)

        # 添加残差连接
        residual = output + residual

        return output,residual


if __name__ == '__main__':
    model = RefUnet(in_ch=7, inc_ch=64, num_classes=7)
    input_tensor = torch.randn(4, 7, 640, 640)  # Batch size = 4, 输入通道 = 7, 分辨率 = 640x640
    output = model(input_tensor)

    print('Input size:', input_tensor.size())
    print('Output size:', output.size())
