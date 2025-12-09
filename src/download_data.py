"""
Script para baixar dataset do Kaggle
Trabalho Final IA
"""

import os
import zipfile
import kaggle
from pathlib import Path


def download_dataset():
    """
    Baixa dataset do Kaggle: Recod.ai/LUC - Scientific Image Forgery Detection
    
    Pré-requisito: Configurar kaggle.json em ~/.kaggle/
    """
    
    competition_name = 'recod-luc-scientific-image-forgery-detection'
    data_dir = Path('./data')
    
    print("="*60)
    print("DOWNLOADING DATASET FROM KAGGLE")
    print("="*60)
    print(f"Competition: {competition_name}")
    print(f"Destination: {data_dir}")
    print()
    
    # Criar diretório
    data_dir.mkdir(exist_ok=True)
    
    # Verificar autenticação Kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated")
    except Exception as e:
        print(f"✗ Error authenticating Kaggle API: {e}")
        print("\nInstructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New Token' in API section")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Download dataset
    print("\nDownloading competition files...")
    try:
        api.competition_download_files(
            competition_name, 
            path=str(data_dir),
            quiet=False
        )
        print("✓ Download completed")
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print("\nMake sure you:")
        print("1. Accepted competition rules at:")
        print(f"   https://www.kaggle.com/c/{competition_name}/rules")
        return
    
    # Extrair arquivos
    print("\nExtracting files...")
    zip_file = data_dir / f'{competition_name}.zip'
    
    if zip_file.exists():
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("✓ Extraction completed")
            
            # Remover zip
            zip_file.unlink()
            print(f"✓ Removed {zip_file}")
            
        except Exception as e:
            print(f"✗ Error extracting: {e}")
            return
    
    # Verificar estrutura
    print("\nVerifying dataset structure...")
    required_dirs = [
        'train_images/authentic',
        'train_images/forged',
        'train_masks',
        'test_images',
    ]
    
    all_present = True
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}")
        else:
            print(f"✗ {dir_name} not found")
            all_present = False
    
    if all_present:
        print("\n" + "="*60)
        print("DATASET READY!")
        print("="*60)
        
        # Estatísticas
        authentic_dir = data_dir / 'train_images/authentic'
        forged_dir = data_dir / 'train_images/forged'
        masks_dir = data_dir / 'train_masks'
        test_dir = data_dir / 'test_images'
        
        print(f"\nDataset statistics:")
        print(f"  Authentic images: {len(list(authentic_dir.glob('*.jpg')))}")
        print(f"  Forged images: {len(list(forged_dir.glob('*.jpg')))}")
        print(f"  Training masks: {len(list(masks_dir.glob('*.npy')))}")
        print(f"  Test images: {len(list(test_dir.glob('*.jpg')))}")
        print()
    else:
        print("\n✗ Dataset structure incomplete")
        print("Please check the download manually")


if __name__ == '__main__':
    download_dataset()
