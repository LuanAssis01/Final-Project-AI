# Scientific Image Forgery Detection
## Trabalho Final - Inteligência Artificial

Projeto de detecção de falsificação em imagens científicas usando Deep Learning.

### 📋 Descrição

Este projeto implementa e compara **3 modelos diferentes** para detecção de copy-move forgery em imagens biomédicas:

1. **SimpleCNN** - CNN simples from scratch
2. **ResNet50Transfer** - Transfer learning com ResNet50 pré-treinado
3. **UNet** - Segmentação para detectar regiões falsificadas

### 🗂️ Estrutura do Projeto

```
ia_the_movie/
├── data/                          # Dataset (baixar do Kaggle)
│   ├── train_images/
│   │   ├── authentic/
│   │   └── forged/
│   ├── train_masks/
│   ├── test_images/
│   └── sample_submission.csv
│
├── src/                           # Código fonte
│   ├── models.py                  # Implementação dos 3 modelos
│   ├── datasets.py                # Dataset loaders e augmentation
│   ├── train.py                   # Script de treinamento
│   ├── evaluate.py                # Script de avaliação
│   ├── utils.py                   # Funções auxiliares
│   └── download_data.py           # Download do dataset do Kaggle
│
├── configs/
│   └── config.yaml                # Configurações do projeto
│
├── notebooks/
│   └── eda.ipynb                  # Análise exploratória
│
├── checkpoints/                   # Modelos salvos
│   ├── simple_cnn/
│   ├── resnet_transfer/
│   └── unet_segmentation/
│
├── results/                       # Resultados, gráficos, métricas
│   ├── simple_cnn/
│   ├── resnet_transfer/
│   └── unet_segmentation/
│
├── docs/                          # Apresentação final
│
├── requirements.txt               # Dependências
└── README.md
```

### 🚀 Setup Inicial

> **⚠️ Usuários Arch Linux**: Veja [INSTALL_ARCH.md](INSTALL_ARCH.md) para instruções específicas!

#### 1. Criar ambiente virtual

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Importante**: O prompt deve mostrar `(venv)` quando ativado.

#### 2. Instalar dependências

```bash
# Atualizar pip primeiro
pip install --upgrade pip

# Instalar todas as dependências
pip install -r requirements.txt
```

**Nota**: A instalação do PyTorch pode demorar alguns minutos (~2GB).

#### 3. Configurar Kaggle API

Baixe suas credenciais do Kaggle:
- Acesse: https://www.kaggle.com/settings/account
- Seção "API" → "Create New Token"
- Salve o arquivo `kaggle.json` em `~/.kaggle/`

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 4. Baixar dataset

```bash
python src/download_data.py
```

### 🎯 Como Usar

#### Treinar os 3 modelos

```bash
# SimpleCNN
python src/train.py --model simple_cnn

# ResNet50 Transfer Learning
python src/train.py --model resnet_transfer

# U-Net Segmentation
python src/train.py --model unet_segmentation
```

#### Avaliar modelos

```bash
# Avaliar modelo específico
python src/evaluate.py --model simple_cnn

# Avaliar todos e comparar
python src/evaluate.py --model all
```

#### Configurações

Edite `configs/config.yaml` para ajustar:
- Tamanho das imagens
- Batch size (ajustar para sua VRAM)
- Learning rate
- Augmentations
- Número de epochs
- etc.

### 📊 Métricas Utilizadas

#### Classificação (SimpleCNN e ResNet)
- **Accuracy** - Acurácia geral
- **Precision** - Precisão para classe "forged"
- **Recall** - Revocação para classe "forged"
- **F1-Score** - Média harmônica de precisão e recall
- **AUC-ROC** - Área sob a curva ROC
- **Confusion Matrix** - Matriz de confusão

#### Segmentação (U-Net)
- **Pixel Accuracy** - Acurácia por pixel
- **IoU (Intersection over Union)** - Jaccard Index
- **Dice Coefficient** - F1-Score para segmentação
- **Precision** - Precisão por pixel
- **Recall** - Recall por pixel

### 🏗️ Arquitetura dos Modelos

#### 1. SimpleCNN
```
Input (3, 512, 512)
  ↓
Conv Block 1 (32 filters)
  ↓
Conv Block 2 (64 filters)
  ↓
Conv Block 3 (128 filters)
  ↓
Conv Block 4 (256 filters)
  ↓
Conv Block 5 (512 filters)
  ↓
Global Average Pooling
  ↓
FC Layers + Dropout
  ↓
Output (2 classes)
```

#### 2. ResNet50Transfer
```
Input (3, 512, 512)
  ↓
ResNet50 Backbone (pré-treinado ImageNet)
  ↓
Custom FC Head
  ↓
Output (2 classes)
```

#### 3. U-Net
```
Input (3, 512, 512)
  ↓
Encoder (ResNet34 pré-treinado)
  ↓
Bottleneck
  ↓
Decoder (upsampling + skip connections)
  ↓
Output (1, 512, 512) - Máscara de segmentação
```

### 💾 Resultados Esperados

Após o treinamento, você terá:

```
results/
├── simple_cnn/
│   ├── training_history.json
│   ├── evaluation_metrics.csv
│   ├── confusion_matrix.png
│   └── logs/
│
├── resnet_transfer/
│   ├── training_history.json
│   ├── evaluation_metrics.csv
│   ├── confusion_matrix.png
│   └── logs/
│
├── unet_segmentation/
│   ├── training_history.json
│   ├── evaluation_metrics.csv
│   ├── segmentation_examples.png
│   └── logs/
│
└── model_comparison.csv  # Comparação final
```

### 📈 Análise Exploratória

Execute o notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

O notebook inclui:
- Estatísticas do dataset
- Distribuição de classes
- Visualização de imagens e máscaras
- Análise de tamanhos e formatos
- Exemplos de augmentation

### 🎓 Para a Apresentação Final

Incluir na apresentação:

1. **Introdução**
   - Problema: Copy-move forgery em imagens científicas
   - Importância: Integridade científica
   - Dataset: Recod.ai/LUC

2. **Metodologia**
   - 3 abordagens diferentes
   - Arquiteturas dos modelos
   - Data augmentation
   - Train/val split

3. **Experimentos**
   - Configurações de treinamento
   - Hiperparâmetros
   - Hardware utilizado

4. **Resultados**
   - Tabela comparativa de métricas
   - Confusion matrices
   - Exemplos de predições
   - Curvas de treinamento

5. **Conclusões**
   - Qual modelo performou melhor?
   - Limitações
   - Trabalhos futuros

6. **Referências**
   - Papers relevantes
   - Dataset
   - Bibliotecas utilizadas

### 🔧 Troubleshooting

#### CUDA Out of Memory
```yaml
# Em config.yaml, reduza:
dataset:
  batch_size: 4  # era 8
  image_size: 256  # era 512

training:
  mixed_precision: true  # Habilitar
```

#### Modelo não converge
```yaml
# Ajuste learning rate:
training:
  learning_rate: 0.0001  # era 0.001
```

### 📚 Referências

- **Dataset**: [Kaggle - Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recod-luc-scientific-image-forgery-detection)
- **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **ResNet**: He et al. "Deep Residual Learning for Image Recognition"
- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch

### 👥 Equipe

[Adicionar nomes dos integrantes do grupo]

---

**Boa sorte no trabalho final! 🚀**
