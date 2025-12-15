import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp
from typing import Dict


class ResidualBlock(nn.Module):
    """Bloco residual para melhor fluxo de gradiente"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    """CNN APRIMORADA com Residual Blocks e SE Attention"""
    
    def __init__(self, num_classes: int = 2, base_filters: int = 32):
        super(SimpleCNN, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(base_filters, base_filters*2, 2)
        self.layer2 = self._make_layer(base_filters*2, base_filters*4, 2)
        self.layer3 = self._make_layer(base_filters*4, base_filters*8, 2)
        self.layer4 = self._make_layer(base_filters*8, base_filters*16, 2)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_filters*16, base_filters*4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*16, 1),
            nn.Sigmoid()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_filters*16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        se_weight = self.se(x)
        x = x * se_weight
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50Transfer(nn.Module):
    """Transfer Learning APRIMORADO com ResNet50"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super(ResNet50Transfer, self).__init__()
        
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class UNet(nn.Module):
    """U-Net para segmentação"""
    
    def __init__(self, encoder_name: str = 'resnet34', 
                 encoder_weights: str = 'imagenet',
                 in_channels: int = 3,
                 classes: int = 1):
        super(UNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, config: Dict) -> nn.Module:
    """Factory function para criar modelos"""
    model_config = config['models'][model_name]
    
    if model_name == 'simple_cnn':
        return SimpleCNN(
            num_classes=model_config['num_classes'],
            base_filters=model_config['base_filters']
        )
    elif model_name == 'resnet_transfer':
        return ResNet50Transfer(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    elif model_name == 'unet_segmentation':
        return UNet(
            encoder_name=model_config['encoder_name'],
            encoder_weights=model_config['encoder_weights'],
            in_channels=model_config['in_channels'],
            classes=model_config['classes']
        )
    else:
        raise ValueError(f"Modelo '{model_name}' não reconhecido")


def count_parameters(model: nn.Module) -> int:
    """
    Conta o número total de parâmetros treináveis do modelo
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        Número de parâmetros treináveis
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
