"""
Dataset classes para Scientific Image Forgery Detection
Trabalho Final IA
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple, Optional, Dict
import cv2


class ForgeryClassificationDataset(Dataset):
    """
    Dataset para classificação binária: authentic vs forged
    Usado para SimpleCNN e ResNet
    """
    
    def __init__(self, 
                 authentic_dir: str, 
                 forged_dir: str,
                 transform: Optional[A.Compose] = None,
                 image_size: int = 512):
        """
        Args:
            authentic_dir: Diretório com imagens autênticas
            forged_dir: Diretório com imagens falsificadas
            transform: Transformações de augmentation
            image_size: Tamanho para redimensionar imagens
        """
        self.image_size = image_size
        self.transform = transform
        
        # Carregar paths de imagens autênticas
        self.authentic_images = sorted(list(Path(authentic_dir).glob('*.jpg')) + 
                                      list(Path(authentic_dir).glob('*.png')))
        
        # Carregar paths de imagens forged
        self.forged_images = sorted(list(Path(forged_dir).glob('*.jpg')) + 
                                   list(Path(forged_dir).glob('*.png')))
        
        # Criar lista completa de imagens e labels
        self.images = self.authentic_images + self.forged_images
        self.labels = [0] * len(self.authentic_images) + [1] * len(self.forged_images)
        
        print(f"Dataset carregado: {len(self.authentic_images)} autênticas, "
              f"{len(self.forged_images)} falsificadas")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Carregar imagem
        img_path = str(self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Aplicar transformações
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


class ForgerySegmentationDataset(Dataset):
    """
    Dataset para segmentação: detectar região falsificada
    Usado para U-Net
    """
    
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 transform: Optional[A.Compose] = None,
                 image_size: int = 512,
                 include_authentic: bool = True):
        """
        Args:
            images_dir: Diretório com imagens (forged)
            masks_dir: Diretório com máscaras (.npy)
            transform: Transformações de augmentation
            image_size: Tamanho para redimensionar
            include_authentic: Se True, inclui imagens autênticas com máscara vazia
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Listar todas as máscaras disponíveis
        mask_files = sorted(list(Path(masks_dir).glob('*.npy')))
        
        self.samples = []
        
        # Adicionar imagens forged com máscaras
        for mask_path in mask_files:
            # Nome do arquivo sem extensão
            img_id = mask_path.stem
            
            # Procurar imagem correspondente
            img_path = self.images_dir / 'forged' / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = self.images_dir / 'forged' / f"{img_id}.png"
            
            if img_path.exists():
                self.samples.append({
                    'image': img_path,
                    'mask': mask_path,
                    'has_forgery': True
                })
        
        # Adicionar imagens autênticas (máscara vazia)
        if include_authentic:
            authentic_dir = self.images_dir / 'authentic'
            if authentic_dir.exists():
                authentic_images = (list(authentic_dir.glob('*.jpg')) + 
                                  list(authentic_dir.glob('*.png')))
                
                for img_path in authentic_images[:len(self.samples)]:  # Balancear
                    self.samples.append({
                        'image': img_path,
                        'mask': None,
                        'has_forgery': False
                    })
        
        print(f"Segmentation dataset: {len(self.samples)} amostras")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Carregar imagem
        image = cv2.imread(str(sample['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar ou criar máscara
        if sample['mask'] is not None:
            mask = np.load(str(sample['mask']))
            # Garantir máscara binária
            mask = (mask > 0).astype(np.float32)
        else:
            # Máscara vazia para imagens autênticas
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Aplicar transformações (augmentation deve aplicar mesma transformação em image e mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Adicionar dimensão de canal à máscara
        mask = mask.unsqueeze(0)
        
        return image, mask


def get_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """
    Cria pipeline de transformações/augmentations
    
    Args:
        config: Dicionário de configuração
        mode: 'train' ou 'val'
    
    Returns:
        Composição de transformações do Albumentations
    """
    aug_config = config['augmentation'][mode]
    image_size = config['dataset']['image_size']
    
    if mode == 'train':
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.VerticalFlip(p=aug_config['vertical_flip']),
            A.Rotate(limit=aug_config['rotation'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness'],
                contrast_limit=aug_config['contrast'],
                p=0.5
            ),
            A.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            ),
            ToTensorV2()
        ]
    else:  # validation
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            ),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Cria dataloaders de treino e validação para classificação
    
    Args:
        config: Dicionário de configuração
    
    Returns:
        train_loader, val_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Paths
    authentic_dir = os.path.join(config['paths']['train_images'], 'authentic')
    forged_dir = os.path.join(config['paths']['train_images'], 'forged')
    
    # Criar dataset completo
    full_dataset = ForgeryClassificationDataset(
        authentic_dir=authentic_dir,
        forged_dir=forged_dir,
        transform=None,
        image_size=config['dataset']['image_size']
    )
    
    # Split train/val
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=1-config['dataset']['train_split'],
        random_state=config['dataset']['seed'],
        stratify=full_dataset.labels
    )
    
    # Criar subsets com transformações apropriadas
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Aplicar transformações
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )
    
    return train_loader, val_loader


def create_segmentation_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Cria dataloaders de treino e validação para segmentação
    
    Args:
        config: Dicionário de configuração
    
    Returns:
        train_loader, val_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Criar dataset completo
    full_dataset = ForgerySegmentationDataset(
        images_dir=config['paths']['train_images'],
        masks_dir=config['paths']['train_masks'],
        transform=None,
        image_size=config['dataset']['image_size']
    )
    
    # Split train/val
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=1-config['dataset']['train_split'],
        random_state=config['dataset']['seed']
    )
    
    # Transformações
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )
    
    return train_loader, val_loader
