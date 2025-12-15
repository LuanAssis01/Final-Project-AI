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
from sklearn.metrics import confusion_matrix

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
        
        # Carregar checkpoint (PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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
                # Métricas de segmentação (por batch)
                batch_metrics = calculate_segmentation_metrics(outputs, targets)
                
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])
                
                # Guardar para visualização / matriz de confusão
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
        
        # Confusion matrix (por imagem)
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
        
        # Salvar métricas num CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.results_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        
        # Juntar alguns batches para visualização
        images_vis = torch.cat(images[:4])
        masks_true_vis = torch.cat(masks_true[:4])
        masks_pred_vis = torch.cat(masks_pred[:4])
        
        vis_path = self.results_dir / 'segmentation_examples.png'
        visualize_segmentation_results(
            images_vis, masks_true_vis, masks_pred_vis,
            num_samples=8,
            save_path=vis_path
        )
        print(f"Visualization saved to {vis_path}")
        
        # -------- Matriz de confusão por pixel (binária) --------
        # Converter logits para probabilidade e aplicar threshold
        threshold = self.config['inference'].get('threshold', 0.5)
        
        masks_true_all = torch.cat(masks_true)      # (N, 1, H, W) ou (N, H, W)
        masks_pred_all = torch.cat(masks_pred)      # (N, 1, H, W)
        
        # Garantir formato (N, H, W)
        if masks_true_all.ndim == 4:
            masks_true_all = masks_true_all.squeeze(1)
        if masks_pred_all.ndim == 4:
            masks_pred_all = masks_pred_all.squeeze(1)
        
        # Aplicar sigmoid se necessário (caso modelo não tenha activation)
        if masks_pred_all.min() < 0 or masks_pred_all.max() > 1:
            masks_pred_probs = torch.sigmoid(masks_pred_all)
        else:
            masks_pred_probs = masks_pred_all
        
        masks_pred_bin = (masks_pred_probs > threshold).int()
        masks_true_bin = (masks_true_all > 0.5).int()
        
        # Achatando para vetor 1D de pixels
        y_true = masks_true_bin.view(-1).cpu().numpy()
        y_pred = masks_pred_bin.view(-1).cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Plotar matriz de confusão de segmentação
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=['Background', 'Forgery'],
            yticklabels=['Background', 'Forgery'],
            ylabel='True label',
            xlabel='Predicted label',
            title='Segmentation Confusion Matrix (per pixel)'
        )
        
        # Anotações
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black'
                )
        
        plt.tight_layout()
        cm_path = self.results_dir / 'segmentation_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Segmentation confusion matrix saved to {cm_path}")
        # --------------------------------------------------------
        
        return metrics
    
    def compare_models(self, model_results: dict):
        """Compara resultados de múltiplos modelos"""
        
        df = pd.DataFrame(model_results).T
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(df.to_string())
        print("="*60 + "\n")
        
        comparison_path = Path(self.config['paths']['results_dir']) / 'model_comparison.csv'
        df.to_csv(comparison_path)
        print(f"Comparison saved to {comparison_path}")
        
        self._plot_model_comparison(df, comparison_path.parent / 'model_comparison.png')
    
    def _plot_model_comparison(self, df: pd.DataFrame, save_path: Path):
        """Plota comparação entre modelos"""
        
        metrics = df.columns.tolist()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
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
        plt.close()


def evaluate_single_model(config_path: str, model_name: str):
    """Avalia um único modelo"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = Path(config['paths']['checkpoints_dir']) / model_name / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
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
    
    if len(results) > 1:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        first_model = list(results.keys())[0]
        evaluator = Evaluator(
            config,
            first_model,
            Path(config['paths']['checkpoints_dir']) / first_model / 'best_model.pth'
        )
        
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
