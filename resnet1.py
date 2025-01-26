import torch.nn as nn
import torch.nn.functional as F
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):  
    expansion = 1 

    def __init__(self, inplanes, planes, stride=1, downsample=None):  
        super(BasicBlock, self).__init__()  
        self.conv1 = conv3x3(inplanes, planes, stride)  
        self.bn1 = nn.BatchNorm2d(planes)  
        self.relu = nn.ReLU(inplace=True)  
        self.conv2 = conv3x3(planes, planes)  
        self.bn2 = nn.BatchNorm2d(planes)  
        self.downsample = downsample  
        self.stride = stride  

    def forward(self, x):  
        residual = x  
        out = self.conv1(x)  
        out = self.bn1(out)  
        out = self.relu(out)  
        out = self.conv2(out)  
        out = self.bn2(out)  
        if self.downsample is not None:  
            residual = self.downsample(x)  
        out += residual  
        out = self.relu(out)  
        return out  
class Bottleneck(nn.Module):  
    expansion = 4  

    def __init__(self, inplanes, planes, stride=1, downsample=None): 
        super(Bottleneck, self).__init__()  
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  
        self.relu = nn.ReLU(inplace=True) 
        self.downsample = downsample  
        self.stride = stride 

    def forward(self, x):  
        residual = x 
        out = self.conv1(x)  
        out = self.bn1(out)  
        out = self.relu(out)  
        out = self.conv2(out)  
        out = self.bn2(out) 
        out = self.relu(out)  
        out = self.conv3(out)  
        out = self.bn3(out)  
        if self.downsample is not None:  
            residual = self.downsample(x)  
        out += residual 
        out = self.relu(out)  
        return out  
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, maps=32, dropout_rate=0.5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
       
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.cbam_first = CBAM(64)  

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(512, maps, 1),
            nn.BatchNorm2d(maps),
            nn.ReLU(inplace=True)
        )
        
        
        self.cbam_last = CBAM(maps) 

        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):  
        downsample = None  
        if stride != 1 or self.inplanes != planes * block.expansion:  
            downsample = nn.Sequential(  
                nn.Conv2d(self.inplanes, planes * block.expansion,  
                          kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [] 
        layers.append(block(self.inplanes, planes, stride, downsample))  
        self.inplanes = planes * block.expansion 
        for i in range(1, blocks): 
            layers.append(block(self.inplanes, planes))  
        return nn.Sequential(*layers)  
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

       
        x = self.cbam_first(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv(x4)

        
        x = self.cbam_last(x)

        x = self.dropout(x)
        return x, x2, x3, x4


def resnet18(pretrained=False, **kwargs):  
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  
    if pretrained:  
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet18'], progress=True))  
    return model  