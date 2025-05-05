import torch
import torch.nn as nn

class FeatureAlignmentModule(nn.Module):
    def __init__(self, in_dim, adapt_dim, kernel_size=3):
        """
        Feature Alignment Module (FAM)
        Args:
            in_dim (int): 输入特征维度 (ViT 输出特征的通道数)。
            adapt_dim (int): 中间特征调整维度。
            kernel_size (int): 卷积核大小，默认为 3。
        """
        super(FeatureAlignmentModule, self).__init__()

        # 1. 特征正则化
        self.norm = nn.LayerNorm(in_dim)

        # 2. 轻量卷积变换
        self.conv1 = nn.Conv2d(in_dim, adapt_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(adapt_dim, in_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        # 3. 残差连接
        self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (B, N, C)，其中
                B: 批量大小
                N: patch 数量
                C: 特征维度
        """
        B, N, C = x.size()

        # 1. 特征正则化
        x = self.norm(x)

        # 2. 转换为卷积适配的形状 (B, C, H, W)
        H = W = int(N ** 0.5)  # 假设 N 是一个平方数
        x = x.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        # 3. 卷积调整
        x_residual = self.residual(x)  # 原始输入作为残差
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        # 4. 加入残差连接
        x = x + x_residual

        # 5. 转换回原始形状 (B, N, C)
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        return x
if __name__ == '__main__':
    # 定义 FeatureAlignmentModule 的输入输出维度
    in_dim = 768  # ViT 编码器的输出维度
    adapt_dim = 384  # 中间调整的特征维度

    # 实例化 FeatureAlignmentModule
    model = FeatureAlignmentModule(in_dim=in_dim, adapt_dim=adapt_dim)

    # 创建一个测试输入张量
    batch_size = 4  # 批量大小
    num_patches = 1600  # 假设每张图像分割为 32x32 个 patch (32x32 = 1024)
    input_tensor = torch.randn(batch_size, num_patches, in_dim)  # 输入形状 (B, N, C)

    # 运行前向传播
    output = model(input_tensor)

    # 打印输入和输出的形状
    print('Input size:', input_tensor.size())  # (B, N, C)
    print('Output size:', output.size())  # (B, N, C)
