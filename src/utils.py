"""
Funções utilitárias: métricas, visualização, logging
Trabalho Final IA
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from typing import Dict, List, Tuple
import os
from pathlib import Path
import json


# ==================== LOSS FUNCTIONS ====================

class DiceLoss(nn.Module):
    """Dice Loss para segmentação"""
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combinação de Dice Loss e Binary Cross Entropy"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions, targets):
        dice_loss = self.dice(predictions, targets)
        bce_loss = self.bce(predictions, targets)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def get_loss_function(loss_name: str, device: str = 'cuda'):
    """Factory para loss functions"""
    
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss().to(device)
    
    elif loss_name == 'dice':
        return DiceLoss().to(device)
    
    elif loss_name == 'dice_bce':
        return DiceBCELoss().to(device)
    
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss().to(device)
    
    else:
        raise ValueError(f"Loss desconhecida: {loss_name}")


# ==================== MÉTRICAS - CLASSIFICAÇÃO ====================

def calculate_classification_metrics(predictions: np.ndarray, 
                                     targets: np.ndarray,
                                     probabilities: np.ndarray = None) -> Dict:
    """
    Calcula métricas de classificação
    
    Args:
        predictions: Predições (classes)
        targets: Labels verdadeiros
        probabilities: Probabilidades (para AUC-ROC)
    
    Returns:
        Dicionário com métricas
    """
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='binary', zero_division=0),
        'recall': recall_score(targets, predictions, average='binary', zero_division=0),
        'f1_score': f1_score(targets, predictions, average='binary', zero_division=0),
    }
    
    # AUC-ROC se probabilidades fornecidas
    if probabilities is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(targets, probabilities)
        except:
            metrics['auc_roc'] = 0.0
    
    return metrics


def print_classification_report(predictions: np.ndarray, 
                                targets: np.ndarray,
                                class_names: List[str] = None):
    """Imprime relatório de classificação detalhado"""
    
    if class_names is None:
        class_names = ['Authentic', 'Forged']
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(targets, predictions, target_names=class_names))


# ==================== MÉTRICAS - SEGMENTAÇÃO ====================

def dice_coefficient(predictions: torch.Tensor, 
                    targets: torch.Tensor, 
                    threshold: float = 0.5,
                    smooth: float = 1.0) -> float:
    """Calcula Dice Coefficient"""
    
    predictions = (predictions > threshold).float()
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()


def iou_score(predictions: torch.Tensor, 
              targets: torch.Tensor,
              threshold: float = 0.5,
              smooth: float = 1e-6) -> float:
    """Calcula IoU (Intersection over Union)"""
    
    predictions = (predictions > threshold).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """Calcula acurácia por pixel"""
    
    predictions = (predictions > threshold).float()
    correct = (predictions == targets).sum()
    total = targets.numel()
    
    return (correct / total).item()


def calculate_segmentation_metrics(predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   threshold: float = 0.5) -> Dict:
    """
    Calcula métricas de segmentação
    
    Args:
        predictions: Predições (logits ou probabilidades)
        targets: Máscaras ground truth
        threshold: Threshold para binarização
    
    Returns:
        Dicionário com métricas
    """
    
    # Aplicar sigmoid se necessário
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    metrics = {
        'pixel_accuracy': pixel_accuracy(predictions, targets, threshold),
        'iou': iou_score(predictions, targets, threshold),
        'dice': dice_coefficient(predictions, targets, threshold),
    }
    
    # Métricas binárias
    preds_flat = (predictions > threshold).view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    
    metrics['precision'] = precision_score(targets_flat, preds_flat, zero_division=0)
    metrics['recall'] = recall_score(targets_flat, preds_flat, zero_division=0)
    
    return metrics


# ==================== VISUALIZAÇÃO ====================

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None):
    """Plota matriz de confusão"""
    
    if class_names is None:
        class_names = ['Authentic', 'Forged']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: Dict, save_path: str = None):
    """Plota curvas de treinamento"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Métrica principal (ex: accuracy ou F1)
    metric_key = 'val_accuracy' if 'val_accuracy' in history else 'val_f1_score'
    metric_name = 'Accuracy' if 'val_accuracy' in history else 'F1 Score'
    
    if metric_key in history:
        axes[1].plot(history[metric_key.replace('val_', 'train_')], label=f'Train {metric_name}')
        axes[1].plot(history[metric_key], label=f'Val {metric_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_segmentation_results(images: torch.Tensor,
                                   masks_true: torch.Tensor,
                                   masks_pred: torch.Tensor,
                                   num_samples: int = 4,
                                   threshold: float = 0.5,
                                   save_path: str = None):
    """
    Visualiza resultados de segmentação
    
    Args:
        images: Imagens originais [B, C, H, W]
        masks_true: Máscaras ground truth [B, 1, H, W]
        masks_pred: Máscaras preditas [B, 1, H, W]
        num_samples: Número de amostras para visualizar
        threshold: Threshold para binarização
        save_path: Caminho para salvar figura
    """
    
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Desnormalizar imagem
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        mask_true = masks_true[i, 0].cpu().numpy()
        mask_pred = torch.sigmoid(masks_pred[i, 0]).cpu().numpy()
        mask_pred_bin = (mask_pred > threshold).astype(float)
        
        # Imagem original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Máscara ground truth
        axes[i, 1].imshow(mask_true, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Máscara predita (probabilidade)
        axes[i, 2].imshow(mask_pred, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Probability')
        axes[i, 2].axis('off')
        
        # Máscara predita (binarizada)
        axes[i, 3].imshow(mask_pred_bin, cmap='gray')
        axes[i, 3].set_title(f'Predicted Mask (t={threshold})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ==================== LOGGING ====================

class MetricsLogger:
    """Logger para salvar métricas durante treinamento"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def update(self, metrics: Dict, epoch: int):
        """Adiciona métricas de uma época"""
        
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def save(self, filename: str = 'training_history.json'):
        """Salva histórico em JSON"""
        
        save_path = self.log_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"Training history saved to {save_path}")
    
    def load(self, filename: str = 'training_history.json'):
        """Carrega histórico de JSON"""
        
        load_path = self.log_dir / filename
        
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.history = json.load(f)
            print(f"Training history loaded from {load_path}")
        else:
            print(f"No history file found at {load_path}")


# ==================== CHECKPOINT ====================

def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   save_path: str):
    """Salva checkpoint do modelo"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str,
                   device: str = 'cuda'):
    """Carrega checkpoint do modelo"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}")
    
    return epoch, metrics


# ==================== UTILS ====================

def set_seed(seed: int):
    """Define seed para reprodutibilidade"""
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """Retorna device (cuda ou cpu)"""
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
