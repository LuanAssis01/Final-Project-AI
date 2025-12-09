"""
Script de treinamento para os 3 modelos
Trabalho Final IA
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from models import get_model, count_parameters
from datasets import create_dataloaders, create_segmentation_dataloaders
from utils import (
    get_loss_function, calculate_classification_metrics, 
    calculate_segmentation_metrics, MetricsLogger,
    save_checkpoint, set_seed, get_device
)


class Trainer:
    """Classe para treinar modelos"""
    
    def __init__(self, config: dict, model_name: str):
        self.config = config
        self.model_name = model_name
        self.device = get_device(config['hardware']['gpu_id'])
        
        # Set seed
        set_seed(config['dataset']['seed'])
        
        # Criar diretórios
        self.checkpoint_dir = Path(config['paths']['checkpoints_dir']) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(config['paths']['results_dir']) / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = MetricsLogger(log_dir=self.results_dir)
        
        # Criar modelo
        self.model = get_model(model_name, config).to(self.device)
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")
        
        # Criar dataloaders
        if model_name == 'unet_segmentation':
            self.train_loader, self.val_loader = create_segmentation_dataloaders(config)
        else:
            self.train_loader, self.val_loader = create_dataloaders(config)
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}\n")
        
        # Loss function
        loss_name = config['training']['loss'][model_name]
        self.criterion = get_loss_function(loss_name, self.device)
        
        # Optimizer
        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']
        
        if config['training']['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, 
                                            momentum=0.9, weight_decay=wd)
        
        # Scheduler
        if config['training']['scheduler'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        elif config['training']['scheduler'] == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['training']['epochs']
            )
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.patience = config['training']['patience']
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Task type
        self.is_segmentation = (model_name == 'unet_segmentation')
    
    def train_epoch(self, epoch: int):
        """Treina uma época"""
        
        self.model.train()
        running_loss = 0.0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['clip_grad_norm']:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['clip_grad_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config['training']['clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['clip_grad_norm']
                    )
                
                self.optimizer.step()
            
            running_loss += loss.item()
            
            # Coletar predições para métricas
            if not self.is_segmentation:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        avg_loss = running_loss / len(self.train_loader)
        
        # Calcular métricas
        metrics = {'train_loss': avg_loss}
        
        if not self.is_segmentation:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            all_probs = np.array(all_probs)
            
            train_metrics = calculate_classification_metrics(all_preds, all_targets, all_probs)
            metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Valida o modelo"""
        
        self.model.eval()
        running_loss = 0.0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        if self.is_segmentation:
            all_iou = []
            all_dice = []
            all_pixel_acc = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # Métricas
            if self.is_segmentation:
                seg_metrics = calculate_segmentation_metrics(outputs, targets)
                all_iou.append(seg_metrics['iou'])
                all_dice.append(seg_metrics['dice'])
                all_pixel_acc.append(seg_metrics['pixel_accuracy'])
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        avg_loss = running_loss / len(self.val_loader)
        
        metrics = {'val_loss': avg_loss}
        
        if self.is_segmentation:
            metrics.update({
                'val_iou': np.mean(all_iou),
                'val_dice': np.mean(all_dice),
                'val_pixel_accuracy': np.mean(all_pixel_acc)
            })
        else:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            all_probs = np.array(all_probs)
            
            val_metrics = calculate_classification_metrics(all_preds, all_targets, all_probs)
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        return metrics
    
    def train(self):
        """Loop de treinamento completo"""
        
        epochs = self.config['training']['epochs']
        metric_for_best = self.config['validation']['metric_for_best']
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Best model will be selected based on: {metric_for_best}\n")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Combinar métricas
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log
            self.logger.update(epoch_metrics, epoch)
            
            # Print
            print(f"\nEpoch {epoch+1}/{epochs}:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Scheduler step
            if self.config['training']['scheduler'] == 'reduce_on_plateau':
                self.scheduler.step(epoch_metrics[f'val_{metric_for_best}'])
            else:
                self.scheduler.step()
            
            # Salvar checkpoint
            current_metric = epoch_metrics[f'val_{metric_for_best}']
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Salvar melhor modelo
                save_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(self.model, self.optimizer, epoch, epoch_metrics, save_path)
                
                print(f"  ✓ New best model! {metric_for_best}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 60)
        
        # Salvar histórico
        self.logger.save()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {metric_for_best}: {self.best_metric:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Scientific Image Forgery Detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       choices=['simple_cnn', 'resnet_transfer', 'unet_segmentation'],
                       help='Model to train')
    
    args = parser.parse_args()
    
    # Carregar configuração
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Treinar
    trainer = Trainer(config, args.model)
    trainer.train()


if __name__ == '__main__':
    main()
