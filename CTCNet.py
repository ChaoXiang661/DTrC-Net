import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import models as resnet_model
from model.deit import deit_tiny_distilled_patch16_256, deit_tiny_patch16_256

class resnet34(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x0 = self.relu(x)
        feature1 = self.layer1(x0)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32

        return x0, feature1, feature2, feature3, feature4

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat = feat + x
        return feat

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        residual = self.shortcut(x)
        out += residual
        return F.relu(out)

class FeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, in_channel, out_channel):
        super(FeatureFusion, self).__init__()
        self.fusion = ResBlock(in_channel, out_channel)

    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_low, x_high), dim=1)
        x = self.fusion(x)

        return x

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class CTCNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CTCNet, self).__init__()
        self.n_class = n_classes
        self.inchannels = n_channels
        size = 256
        self.cnn = resnet34(pretrained=True)
        self.headpool = PyramidPoolingModule()

        transformer = deit_tiny_distilled_patch16_256()
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )
        self.se = SEBlock(channel=512)
        self.se1 = SEBlock(channel=64)
        self.se2 = SEBlock(channel=128)
        self.se3 = SEBlock(channel=256)

        self.fusion = FeatureFusion(in_channel=512 + size, out_channel=512)
        self.fusion1 = FeatureFusion(in_channel=64 + size, out_channel=64)
        self.fusion2 = FeatureFusion(in_channel=128 + size, out_channel=128)
        self.fusion3 = FeatureFusion(in_channel=256 + size, out_channel=256)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        e0, e1, e2, e3, e4 = self.cnn(x)
        feature_cnn = self.headpool(e4)
        size = 256

        emb = self.patch_embed(x)
        emb = self.transformers[0](emb)
        emb = self.transformers[1](emb)
        emb1 = self.transformers[2](emb)
        feature_tf1 = emb1.permute(0, 2, 1)
        feature_tf1 = feature_tf1.view(b, size, 16, 16)

        emb1 = self.transformers[3](emb1)
        emb1 = self.transformers[4](emb1)
        emb2 = self.transformers[5](emb1)
        feature_tf2 = emb2.permute(0, 2, 1)
        feature_tf2 = feature_tf2.view(b, size, 16, 16)

        emb2 = self.transformers[6](emb2)
        emb2 = self.transformers[7](emb2)
        emb3 = self.transformers[8](emb2)
        feature_tf3 = emb3.permute(0, 2, 1)
        feature_tf3 = feature_tf3.view(b, size, 16, 16)

        emb3 = self.transformers[9](emb3)
        emb3 = self.transformers[10](emb3)
        emb4 = self.transformers[11](emb3)

        feature_tf = emb4.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, size, 16, 16)

        feature_cat = self.fusion(feature_cnn, feature_tf)
        feature_cat = self.se(feature_cat)

        e1 = self.fusion1(e1, feature_tf1)
        e2 = self.fusion2(e2, feature_tf2)
        e3 = self.fusion3(e3, feature_tf3)
        e1 = self.se1(e1)
        e2 = self.se2(e2)
        e3 = self.se3(e3)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)

        d4 = self.decoder4(feature_cat) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out_0 = self.final_conv3(out)
        out = torch.sigmoid(out_0)
        return out_0, out




