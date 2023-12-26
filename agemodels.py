import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义网络结构
# 第二步网络AgeNet
class AgeNet(nn.Module):
    def __init__(self):
        super(AgeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class AgeNet_gray(nn.Module):
    def __init__(self):
        super(AgeNet_gray, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# 端对端网络Unet
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


def conv_block_reduce(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器（下采样）
        self.enc_conv1 = conv_block(3, 64)
        self.enc_conv2 = conv_block(64, 128)
        self.enc_conv3 = conv_block(128, 256)
        self.enc_conv4 = conv_block(256, 512)

        # 中间层
        self.middle_conv1 = conv_block(512, 1024)
        self.middle_conv2 = conv_block(1024, 512)

        # 解码器（上采样）
        self.dec_conv4 = conv_block(1024, 256)
        self.dec_conv3 = conv_block(512, 128)
        self.dec_conv2 = conv_block(256, 64)
        self.dec_conv1 = conv_block(128, 64)

        # 输出层
        self.reduce_size_conv = conv_block_reduce(64, 64)
        self.reduce_size_pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(F.max_pool2d(enc1, 2))
        enc3 = self.enc_conv3(F.max_pool2d(enc2, 2))
        enc4 = self.enc_conv4(F.max_pool2d(enc3, 2))

        # 中间层
        middle = self.middle_conv1(F.max_pool2d(enc4, 2))
        middle = self.middle_conv2(middle)

        # 解码器
        dec4 = F.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=False)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.dec_conv1(dec1)

        # 输出层
        output = self.reduce_size_conv(dec1)
        output = self.reduce_size_conv(output)
        output = self.reduce_size_pool(output)
        output = self.final_conv(output)

        return output.squeeze()
