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


def load_and_fix_mask(mask_path: str) -> np.ndarray:
    """
    Carrega e corrige máscaras com shapes inconsistentes
    
    Args:
        mask_path: Caminho para o arquivo .npy da máscara
        
    Returns:
        Máscara 2D de shape (H, W)
    """
    try:
        mask = np.load(mask_path)
        
        # Se a máscara tem 3 dimensões
        if mask.ndim == 3:
            # Para máscaras multi-canal, usa max() para combinar
            if mask.shape[0] in [1, 2, 3, 4]:  # canais na primeira dimensão
                mask = mask.max(axis=0)  # (C, H, W) -> (H, W)
            elif mask.shape[2] in [1, 2, 3, 4]:  # canais na última dimensão
                mask = mask.max(axis=2)  # (H, W, C) -> (H, W)
            else:
                # Fallback: usar squeeze para remover dimensões de tamanho 1
                mask = np.squeeze(mask)
        
        # Garantir que é 2D
        while mask.ndim > 2:
            mask = mask.max(axis=0)
        
        # Binarizar a máscara (0 ou 1)
        mask = (mask > 0).astype(np.float32)
        
        return mask
        
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        # Retorna máscara vazia em caso de erro
        return np.zeros((256, 256), dtype=np.float32)


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

        self.authentic_images = sorted(list(Path(authentic_dir).glob('*.jpg')) + 
                                      list(Path(authentic_dir).glob('*.png')))
        
        self.forged_images = sorted(list(Path(forged_dir).glob('*.jpg')) + 
                                   list(Path(forged_dir).glob('*.png')))

        self.images = self.authentic_images + self.forged_images
        self.labels = [0] * len(self.authentic_images) + [1] * len(self.forged_images)
        
        print(f"Dataset carregado: {len(self.authentic_images)} autênticas, "
              f"{len(self.forged_images)} falsificadas")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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
        
        self.samples = []
        skipped = 0
        
        # CORRIGIDO: Definir mask_files primeiro
        mask_files = sorted(list(Path(masks_dir).glob('*.npy')))
        
        # CORRIGIDO: Agora o loop funciona corretamente
        for mask_path in mask_files:
            img_id = mask_path.stem

            img_path = self.images_dir / 'forged' / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = self.images_dir / 'forged' / f"{img_id}.png"
            
            if img_path.exists():
                # Verificar se a imagem pode ser carregada
                test_img = cv2.imread(str(img_path))
                if test_img is not None:
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'has_forgery': True
                    })
                else:
                    skipped += 1
        
        if include_authentic:
            authentic_dir = self.images_dir / 'authentic'
            if authentic_dir.exists():
                authentic_images = (list(authentic_dir.glob('*.jpg')) + 
                                  list(authentic_dir.glob('*.png')))
                
                for img_path in authentic_images[:len(self.samples)]:  # Balancear
                    # Verificar se a imagem pode ser carregada
                    test_img = cv2.imread(str(img_path))
                    if test_img is not None:
                        self.samples.append({
                            'image': img_path,
                            'mask': None,
                            'has_forgery': False
                        })
                    else:
                        skipped += 1
        
        print(f"Segmentation dataset: {len(self.samples)} amostras")
        if skipped > 0:
            print(f"  (Skipped {skipped} corrupted/unreadable images)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Carregar imagem
        image = cv2.imread(str(sample['image']))
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # CORRIGIDO: Usar load_and_fix_mask para carregar máscaras
        if sample['mask'] is not None:
            try:
                mask = load_and_fix_mask(str(sample['mask']))
                
                # Redimensionar máscara para o tamanho da imagem se necessário
                if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                    if image.shape[0] > 0 and image.shape[1] > 0:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                    else:
                        raise ValueError(f"Invalid image dimensions: {image.shape}")
            except Exception as e:
                print(f"Error processing mask {sample['mask']}: {e}")
                # Criar máscara vazia em caso de erro
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            # Máscara vazia para imagens autênticas
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.unsqueeze(0)
        
        return image, mask


def get_transforms(config: Dict, mode: str = 'train') -> A.Compose:
    """
    Cria pipeline de transformações/augmentations APRIMORADO
    
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
            A.Rotate(limit=aug_config['rotation'], p=0.6),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness'],
                contrast_limit=aug_config['contrast'],
                p=0.6
            ),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(
                mean=aug_config['normalize']['mean'],
                std=aug_config['normalize']['std']
            ),
            ToTensorV2()
        ]
    else:  
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
    
    authentic_dir = os.path.join(config['paths']['train_images'], 'authentic')
    forged_dir = os.path.join(config['paths']['train_images'], 'forged')
    
    full_dataset = ForgeryClassificationDataset(
        authentic_dir=authentic_dir,
        forged_dir=forged_dir,
        transform=None,
        image_size=config['dataset']['image_size']
    )
    
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=1-config['dataset']['train_split'],
        random_state=config['dataset']['seed'],
        stratify=full_dataset.labels
    )
    
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
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
    
    full_dataset = ForgerySegmentationDataset(
        images_dir=config['paths']['train_images'],
        masks_dir=config['paths']['train_masks'],
        transform=None,
        image_size=config['dataset']['image_size']
    )
    
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=1-config['dataset']['train_split'],
        random_state=config['dataset']['seed']
    )
    
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
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
