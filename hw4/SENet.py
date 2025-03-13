import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBottleneck(Bottleneck):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__(in_channels, out_channels, stride, downsample)
        self.se_block = SEBlock(self.conv3.out_channels, reduction)

    def forward(self, x):
        # Save the identity (residual) connection
        identity = x

        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply SE block on the main branch output
        out = self.se_block(out)

        # Downsample the input if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual connection
        out += identity
        out = self.relu(out)

        return out

def se_resnet50(pretrained=True, reduction=16):
    model = models.resnet50(pretrained=pretrained)
    # Replace Bottleneck with SEBottleneck in all layers
    model.layer1 = nn.Sequential(
        *[SEBottleneck(
            in_channels=block.conv1.in_channels,
            out_channels=block.conv3.out_channels,
            stride=block.stride,
            downsample=block.downsample,
            reduction=reduction
          ) for block in model.layer1]
    )
    model.layer2 = nn.Sequential(
        *[SEBottleneck(
            in_channels=block.conv1.in_channels,
            out_channels=block.conv3.out_channels,
            stride=block.stride,
            downsample=block.downsample,
            reduction=reduction
          ) for block in model.layer2]
    )
    model.layer3 = nn.Sequential(
        *[SEBottleneck(
            in_channels=block.conv1.in_channels,
            out_channels=block.conv3.out_channels,
            stride=block.stride,
            downsample=block.downsample,
            reduction=reduction
          ) for block in model.layer3]
    )
    model.layer4 = nn.Sequential(
        *[SEBottleneck(
            in_channels=block.conv1.in_channels,
            out_channels=block.conv3.out_channels,
            stride=block.stride,
            downsample=block.downsample,
            reduction=reduction
          ) for block in model.layer4]
    )
    return model

model = se_resnet50(pretrained=True, reduction=16)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 7),
    nn.LogSoftmax(dim=1)
)
print(model)