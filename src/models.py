"""
Implementação dos 3 modelos para comparação:
1. SimpleCNN - CNN simples from scratch
2. ResNet50Transfer - Transfer learning com ResNet50
3. UNet - Segmentação para detectar regiões falsificadas

Trabalho Final IA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp
from typing import Dict


class SimpleCNN(nn.Module):
    """
    CNN simples from scratch para classificação binária
    Arquitetura: Conv layers + Pooling + FC layers
    """
    
    def __init__(self, num_classes: int = 2, base_filters: int = 32):
        super(SimpleCNN, self).__init__()
        
        # Encoder (feature extraction)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 512 -> 256
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*8, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*16, base_filters*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_filters*16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResNet50Transfer(nn.Module):
    """
    Transfer Learning usando ResNet50 pré-treinado
    Fine-tuning para classificação binária
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, 
                 freeze_backbone: bool = False):
        super(ResNet50Transfer, self).__init__()
        
        # Carregar ResNet50 pré-treinado no ImageNet
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Congelar backbone se necessário
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Substituir última camada FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Descongelar backbone para fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class UNet(nn.Module):
    """
    U-Net para segmentação de regiões falsificadas
    Usa segmentation_models_pytorch com encoder pré-treinado
    """
    
    def __init__(self, 
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 in_channels: int = 3,
                 classes: int = 1):
        super(UNet, self).__init__()
        
        # Criar U-Net com encoder pré-treinado
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # Sigmoid será aplicado na loss
        )
    
    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, config: Dict) -> nn.Module:
    """
    Factory function para criar modelos
    
    Args:
        model_name: 'simple_cnn', 'resnet_transfer', ou 'unet_segmentation'
        config: Dicionário de configuração
    
    Returns:
        Modelo PyTorch
    """
    model_config = config['models'][model_name]
    
    if model_name == 'simple_cnn':
        model = SimpleCNN(
            num_classes=model_config['num_classes'],
            base_filters=model_config['base_filters']
        )
    
    elif model_name == 'resnet_transfer':
        model = ResNet50Transfer(
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config['freeze_backbone']
        )
    
    elif model_name == 'unet_segmentation':
        model = UNet(
            encoder_name=model_config['encoder_name'],
            encoder_weights=model_config['encoder_weights'],
            in_channels=model_config['in_channels'],
            classes=model_config['classes']
        )
    
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Conta número de parâmetros treináveis"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 512, 512)):
    """
    Imprime resumo do modelo
    
    Args:
        model: Modelo PyTorch
        input_size: Tamanho do input (batch, channels, height, width)
    """
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Testar forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Erro no forward pass: {e}")


if __name__ == '__main__':
    # Teste dos modelos
    print("=" * 60)
    print("TESTANDO MODELOS")
    print("=" * 60)
    
    # SimpleCNN
    print("\n1. SimpleCNN")
    model1 = SimpleCNN(num_classes=2, base_filters=32)
    get_model_summary(model1)
    
    # ResNet50Transfer
    print("\n2. ResNet50Transfer")
    model2 = ResNet50Transfer(num_classes=2, pretrained=False)
    get_model_summary(model2)
    
    # UNet
    print("\n3. UNet")
    model3 = UNet(encoder_name='resnet34', encoder_weights=None)
    get_model_summary(model3)
