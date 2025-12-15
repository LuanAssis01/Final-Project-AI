"""
Script de treinamento para os 3 modelos
Trabalho Final IA
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import time
from datetime import datetime, timedelta

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

        set_seed(config['dataset']['seed'])

        self.checkpoint_dir = Path(config['paths']['checkpoints_dir']) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path(config['paths']['results_dir']) / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = MetricsLogger(log_dir=self.results_dir)

        self.model = get_model(model_name, config).to(self.device)
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        if model_name == 'unet_segmentation':
            self.train_loader, self.val_loader = create_segmentation_dataloaders(config)
        else:
            self.train_loader, self.val_loader = create_dataloaders(config)

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}\n")

        loss_name = config['training']['loss'][model_name]
        self.criterion = get_loss_function(loss_name, self.device, config)

        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']

        if config['training']['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif config['training']['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=0.9, weight_decay=wd
            )

        if config['training']['scheduler'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        elif config['training']['scheduler'] == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['training']['epochs']
            )
        else:
            self.scheduler = None

        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.patience = config['training']['patience']
        self.best_metric = 0.0
        self.patience_counter = 0

        self.is_segmentation = (model_name == 'unet_segmentation')

        # Cronometragem
        self.epoch_times = []
        self.total_start_time = None

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

            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

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

            if not self.is_segmentation:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)

                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

        avg_loss = running_loss / len(self.train_loader)

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
        
        # Ajustar métrica para segmentação se necessário
        if self.is_segmentation:
            # Mapeamento de métricas equivalentes
            metric_mapping = {
                'f1_score': 'dice',  # Dice é equivalente ao F1 para segmentação
                'accuracy': 'pixel_accuracy'
            }
            
            if metric_for_best in metric_mapping:
                original_metric = metric_for_best
                metric_for_best = metric_mapping[metric_for_best]
                print(f"Note: Using '{metric_for_best}' for segmentation (equivalent to '{original_metric}')")
            
            # Validar que a métrica existe para segmentação
            valid_seg_metrics = ['iou', 'dice', 'pixel_accuracy']
            if metric_for_best not in valid_seg_metrics:
                print(f"Warning: '{metric_for_best}' not available for segmentation. Using 'dice'.")
                metric_for_best = 'dice'

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Best model will be selected based on: {metric_for_best}")
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.total_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            epoch_metrics = {**train_metrics, **val_metrics}

            self.logger.update(epoch_metrics, epoch)

            print(f"\nEpoch {epoch+1}/{epochs} - Time: {timedelta(seconds=int(epoch_time))}")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Scheduler
            if self.scheduler is not None:
                if self.config['training']['scheduler'] == 'reduce_on_plateau':
                    metric_key = f'val_{metric_for_best}'
                    if metric_key in epoch_metrics:
                        self.scheduler.step(epoch_metrics[metric_key])
                    else:
                        print(f"Warning: Metric '{metric_key}' not found in epoch_metrics")
                else:
                    self.scheduler.step()

            # Obter métrica atual com tratamento de erro
            metric_key = f'val_{metric_for_best}'
            current_metric = epoch_metrics.get(metric_key, 0.0)
            
            if current_metric == 0.0 and metric_key not in epoch_metrics:
                print(f"Warning: Metric '{metric_key}' not found. Available metrics: {list(epoch_metrics.keys())}")

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0

                # salvar melhor modelo usando utils.save_checkpoint
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'

                metrics_for_checkpoint = {
                    metric_for_best: self.best_metric,
                    'val_loss': epoch_metrics['val_loss'],
                    'epoch': epoch,
                }

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics_for_checkpoint,
                    save_path=str(checkpoint_path),
                )

                print(f"  Best model saved! {metric_for_best}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            # Estimativa de tempo restante
            if len(self.epoch_times) > 0:
                avg_epoch_time = np.mean(self.epoch_times)
                remaining_epochs = epochs - (epoch + 1)
                estimated_time = timedelta(seconds=int(avg_epoch_time * remaining_epochs))
                print(f"  Estimated time remaining: {estimated_time}")

            print("-" * 60)

        total_time = time.time() - self.total_start_time

        self.logger.save()

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best {metric_for_best}: {self.best_metric:.4f}")
        print(f"Total training time: {timedelta(seconds=int(total_time))}")
        print(f"Average time per epoch: {timedelta(seconds=int(np.mean(self.epoch_times)))}")
        print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Scientific Image Forgery Detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['simple_cnn', 'resnet_transfer', 'unet_segmentation'],
                        help='Model to train')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, args.model)
    trainer.train()


if __name__ == '__main__':
    main()
