"""
Scientific Image Forgery Detection
Trabalho Final - Inteligência Artificial
"""

__version__ = '1.0.0'
__author__ = 'Equipe IA'

from .models import SimpleCNN, ResNet50Transfer, UNet, get_model
from .datasets import (
    ForgeryClassificationDataset,
    ForgerySegmentationDataset,
    create_dataloaders,
    create_segmentation_dataloaders,
    get_transforms
)
from .utils import (
    DiceLoss,
    DiceBCELoss,
    get_loss_function,
    calculate_classification_metrics,
    calculate_segmentation_metrics,
    plot_confusion_matrix,
    plot_training_history,
    visualize_segmentation_results,
    MetricsLogger,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
    count_parameters
)

__all__ = [
    # Models
    'SimpleCNN',
    'ResNet50Transfer',
    'UNet',
    'get_model',
    
    # Datasets
    'ForgeryClassificationDataset',
    'ForgerySegmentationDataset',
    'create_dataloaders',
    'create_segmentation_dataloaders',
    'get_transforms',
    
    # Utils
    'DiceLoss',
    'DiceBCELoss',
    'get_loss_function',
    'calculate_classification_metrics',
    'calculate_segmentation_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'visualize_segmentation_results',
    'MetricsLogger',
    'save_checkpoint',
    'load_checkpoint',
    'set_seed',
    'get_device',
    'count_parameters',
]
