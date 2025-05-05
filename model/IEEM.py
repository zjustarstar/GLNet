import torch
from torch import nn


class EdgeEnhancer(nn.Module):  # 边缘增强模块
    def __init__(self, in_dim, norm, act):  # 初始化函数，接收输入维度、归一化层和激活函数
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),  # 非对称卷积 3x1
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),  # 非对称卷积 1x3
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class InteractionMEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.width = width  # 宽度表示尺度数量
        self.hidden_dim = hidden_dim

        # 输入卷积（替换为非对称卷积）
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),  # 非对称卷积 3x1
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),  # 非对称卷积 1x3
            norm(hidden_dim),
            nn.Sigmoid()
        )

        # 池化层
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        # 中间卷积层和边缘增强模块
        self.mid_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),  # 非对称卷积 3x1
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),  # 非对称卷积 1x3
                norm(hidden_dim),
                act()
            ) for _ in range(width - 1)
        ])

        self.edge_enhance = nn.ModuleList([
            EdgeEnhancer(hidden_dim, norm, act) for _ in range(width - 1)
        ])

        # 两个交互后的卷积层（替换为非对称卷积）
        self.interaction_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),  # 非对称卷积 3x1
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),  # 非对称卷积 1x3
                norm(hidden_dim),
                act()
            ) for _ in range(2)
        ])

        # 最终输出卷积层（替换为非对称卷积）
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, in_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),  # 非对称卷积 3x1
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),  # 非对称卷积 1x3
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)  # 初始卷积处理
        features = [mid]  # 存储多尺度特征
        residual = mid  # 保存 residual 供后续使用

        # 逐层进行池化、中间卷积和边缘增强
        for i in range(self.width - 1):
            mid = self.pool(mid)  # 池化操作
            mid = self.mid_conv[i](mid)  # 中间卷积
            mid = self.edge_enhance[i](mid)  # 边缘增强
            features.append(mid)  # 保存不同尺度特征

        # 进行交互操作
        cat_12 = torch.cat([features[0], features[1]], dim=1)  # 第一层和第二层特征交互
        cat_23 = torch.cat([features[1], features[2]], dim=1)  # 第二层和第三层特征交互

        # 分别经过单独的卷积层处理
        inter_12 = self.interaction_convs[0](cat_12)
        inter_23 = self.interaction_convs[1](cat_23)

        # 将交互后的两个特征相加，再与 residual 做残差连接
        fused = (inter_12 + inter_23) + residual

        # 最终输出卷积层
        out = self.out_conv(fused)

        return out


if __name__ == '__main__':
    print("---- 测试修正后的 InteractionMEEM ----")
    model = InteractionMEEM(in_dim=64, hidden_dim=32)
    input_tensor = torch.randn(1, 64, 128, 128)  # 输入数据
    output = model(input_tensor)  # 前向传播
    print("Input size:", input_tensor.size())
    print("Output size:", output.size())  # 输出数据的尺寸
