"""
Script de avaliação e geração de resultados
Trabalho Final IA
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import get_model
from datasets import create_dataloaders, create_segmentation_dataloaders
from utils import (
    calculate_classification_metrics, calculate_segmentation_metrics,
    plot_confusion_matrix, print_classification_report,
    visualize_segmentation_results, load_checkpoint, get_device
)


class Evaluator:
    """Classe para avaliar modelos treinados"""
    
    def __init__(self, config: dict, model_name: str, checkpoint_path: str):
        self.config = config
        self.model_name = model_name
        self.device = get_device(config['hardware']['gpu_id'])
        
        # Criar modelo
        self.model = get_model(model_name, config).to(self.device)
        
        # Carregar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Checkpoint metrics: {checkpoint['metrics']}\n")
        
        # Criar dataloaders
        if model_name == 'unet_segmentation':
            _, self.val_loader = create_segmentation_dataloaders(config)
            self.is_segmentation = True
        else:
            _, self.val_loader = create_dataloaders(config)
            self.is_segmentation = False
        
        # Results dir
        self.results_dir = Path(config['paths']['results_dir']) / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self):
        """Avalia modelo no conjunto de validação"""
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        if self.is_segmentation:
            all_images = []
            all_masks_true = []
            all_masks_pred = []
            all_metrics = {
                'iou': [],
                'dice': [],
                'pixel_accuracy': [],
                'precision': [],
                'recall': []
            }
        
        print("Evaluating model...")
        
        for images, targets in tqdm(self.val_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
            
            if self.is_segmentation:
                # Métricas de segmentação
                batch_metrics = calculate_segmentation_metrics(outputs, targets)
                
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])
                
                # Salvar para visualização
                all_images.append(images.cpu())
                all_masks_true.append(targets.cpu())
                all_masks_pred.append(outputs.cpu())
            
            else:
                # Métricas de classificação
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Processar resultados
        if self.is_segmentation:
            return self._process_segmentation_results(
                all_metrics, all_images, all_masks_true, all_masks_pred
            )
        else:
            return self._process_classification_results(
                np.array(all_preds), np.array(all_targets), np.array(all_probs)
            )
    
    def _process_classification_results(self, preds, targets, probs):
        """Processa resultados de classificação"""
        
        # Calcular métricas
        metrics = calculate_classification_metrics(preds, targets, probs)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS - CLASSIFICATION")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("="*60 + "\n")
        
        # Classification report
        print_classification_report(preds, targets)
        
        # Confusion matrix
        cm_path = self.results_dir / 'confusion_matrix.png'
        plot_confusion_matrix(targets, preds, save_path=cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Salvar métricas em CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.results_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        
        return metrics
    
    def _process_segmentation_results(self, all_metrics, images, masks_true, masks_pred):
        """Processa resultados de segmentação"""
        
        # Calcular métricas médias
        metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS - SEGMENTATION")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("="*60 + "\n")
        
        # Salvar métricas
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.results_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        
        # Visualizar exemplos
        images = torch.cat(images[:4])  # Primeiras 4 batches
        masks_true = torch.cat(masks_true[:4])
        masks_pred = torch.cat(masks_pred[:4])
        
        vis_path = self.results_dir / 'segmentation_examples.png'
        visualize_segmentation_results(
            images, masks_true, masks_pred,
            num_samples=8,
            save_path=vis_path
        )
        print(f"Visualization saved to {vis_path}")
        
        return metrics
    
    def compare_models(self, model_results: dict):
        """Compara resultados de múltiplos modelos"""
        
        # Criar tabela comparativa
        df = pd.DataFrame(model_results).T
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(df.to_string())
        print("="*60 + "\n")
        
        # Salvar
        comparison_path = Path(self.config['paths']['results_dir']) / 'model_comparison.csv'
        df.to_csv(comparison_path)
        print(f"Comparison saved to {comparison_path}")
        
        # Gráfico de barras
        self._plot_model_comparison(df, comparison_path.parent / 'model_comparison.png')
    
    def _plot_model_comparison(self, df: pd.DataFrame, save_path: Path):
        """Plota comparação entre modelos"""
        
        metrics = df.columns.tolist()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            df[metric].plot(kind='bar', ax=axes[i], color='steelblue')
            axes[i].set_title(metric.upper())
            axes[i].set_ylabel('Score')
            axes[i].set_ylim([0, 1])
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()


def evaluate_single_model(config_path: str, model_name: str):
    """Avalia um único modelo"""
    
    # Carregar config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Path do checkpoint
    checkpoint_path = Path(config['paths']['checkpoints_dir']) / model_name / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # Avaliar
    evaluator = Evaluator(config, model_name, checkpoint_path)
    metrics = evaluator.evaluate()
    
    return metrics


def evaluate_all_models(config_path: str):
    """Avalia todos os 3 modelos e compara"""
    
    model_names = ['simple_cnn', 'resnet_transfer', 'unet_segmentation']
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}\n")
        
        metrics = evaluate_single_model(config_path, model_name)
        
        if metrics:
            results[model_name] = metrics
    
    # Comparar resultados
    if len(results) > 1:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        evaluator = Evaluator(config, list(results.keys())[0], 
                            Path(config['paths']['checkpoints_dir']) / 
                            list(results.keys())[0] / 'best_model.pth')
        
        evaluator.compare_models(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Scientific Image Forgery Detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='all',
                       choices=['simple_cnn', 'resnet_transfer', 'unet_segmentation', 'all'],
                       help='Model to evaluate')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        evaluate_all_models(args.config)
    else:
        evaluate_single_model(args.config, args.model)


if __name__ == '__main__':
    main()
