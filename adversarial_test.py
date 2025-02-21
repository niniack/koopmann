import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from koopmann.models import Autoencoder, MLP
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import json
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from koopmann.data import MNISTDataset, DatasetConfig

def setup_logger(results_path):
    """Set up logging configuration"""
    # Create logs directory
    log_dir = results_path / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"adversarial_test_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def fgsm_attack(model, loss_fn, data, epsilon, target):
    """
    FGSM attack implementation
    """
    # Make data require gradients
    data.requires_grad = True
    
    # Forward pass
    outputs = model(data)
    
    # Calculate loss
    loss = loss_fn(outputs, target)
    
    # Get gradients
    loss.backward()
    
    # Collect gradients
    data_grad = data.grad.data
    
    # Create perturbation
    sign_data_grad = data_grad.sign()
    
    # Create adversarial example
    perturbed_data = data + epsilon * sign_data_grad
    
    # Clamp to ensure valid pixel range [0,1]
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

def visualize_results(results, save_path):
    """Visualize the results"""
    # Create accuracy comparison plot
    plt.figure(figsize=(10, 6))
    accuracies = [
        results["original_accuracy"],
        results["adversarial_accuracy"],
        results["autoencoder_accuracy"]
    ]
    labels = ["Original", "Adversarial", "Autoencoder Protected"]
    
    sns.barplot(x=labels, y=accuracies)
    plt.title(f"Accuracy Comparison (ε={results['epsilon']})")
    plt.ylabel("Accuracy (%)")
    plt.savefig(save_path / f"accuracy_comparison_epsilon_{results['epsilon']}.png")
    plt.close()

def test_autoencoder_robustness(args):
    # 设置路径
    data_root = Path("/scratch/gs4133/datasets")
    model_path = Path("/scratch/gs4133/model_saves")  # 修改为正确的模型路径
    results_path = Path("/scratch/gs4133/results")
    vis_path = results_path / "visualizations"
    
    # Create directories
    for path in [data_root, results_path, vis_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(results_path)
    
    try:
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 创建数据集配置
        dataset_config = DatasetConfig(
            dataset_name="MNISTDataset",
            num_samples=args.n_samples,
            split="test",
            torch_transform=None,
            seed=42
        )
        
        logger.info(f"Using data from: {data_root}")
        
        # 检查并创建 processed 目录
        processed_dir = data_root / "MNISTDataset" / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # 加载数据集
        try:
            test_dataset = MNISTDataset(
                config=dataset_config,
                root=str(data_root),
                seed=42,
            )
            logger.info(f"Successfully loaded dataset with {len(test_dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Load models
        try:
            # modify the file path
            autoencoder_path = model_path / "k_1_dim_1024_loc_0_autoencoder_mnist_model.safetensors"
            mlp_path = model_path / "mnist_probed.safetensors"

            print(model_path)
            
            # check file path exist?
            if not autoencoder_path.exists():
                logger.error(f"Autoencoder model not found at: {autoencoder_path}")
                logger.info("Please ensure the model file is in the correct location")
                raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_path}")
            
            if not mlp_path.exists():
                logger.error(f"MLP model not found at: {mlp_path}")
                logger.info("Please ensure the model file is in the correct location")
                raise FileNotFoundError(f"MLP model not found: {mlp_path}")
            
            logger.info(f"Loading autoencoder from: {autoencoder_path}")
            autoencoder, ae_metadata = Autoencoder.load_model(autoencoder_path)
            autoencoder = autoencoder.to(device)
            autoencoder.eval()
            
            logger.info(f"Loading MLP from: {mlp_path}")
            mlp = MLP(
                input_dimension=28*28,
                output_dimension=10,
                config=[128],
                nonlinearity="relu"
            )
            mlp.load_model(mlp_path)
            mlp = mlp.to(device)
            mlp.eval()
            
            # Create latent adapter layer
            latent_adapter = nn.Linear(
                in_features=autoencoder.latent_dimension,
                out_features=128
            ).to(device)
            
            # Test parameters
            epsilon = args.epsilon
            n_samples = args.n_samples
            
            original_correct = 0
            adversarial_correct = 0
            autoencoder_robust = 0
            
            # Run tests
            for i, (data, target) in enumerate(test_loader):
                if i >= n_samples:
                    break
                    
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                # Generate adversarial example
                perturbed_data = fgsm_attack(mlp, nn.CrossEntropyLoss(), data, epsilon, target)
                
                with torch.no_grad():
                    # Test original MLP
                    original_output = mlp(data)
                    if original_output.argmax(1) == target:
                        original_correct += 1
                    
                    adv_output = mlp(perturbed_data)
                    if adv_output.argmax(1) == target:
                        adversarial_correct += 1
                    
                    # Test with autoencoder's latent representation
                    latent = autoencoder.encoder(perturbed_data)
                    hidden_features = mlp.modules[0](perturbed_data)
                    adapted_latent = latent_adapter(latent)
                    combined_features = hidden_features + adapted_latent
                    ae_output = mlp.modules[-1](combined_features)
                    
                    if ae_output.argmax(1) == target:
                        autoencoder_robust += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f'Testing sample {i+1}/{n_samples}')
            
            # Save results
            results = {
                "original_accuracy": 100 * original_correct / n_samples,
                "adversarial_accuracy": 100 * adversarial_correct / n_samples,
                "autoencoder_accuracy": 100 * autoencoder_robust / n_samples,
                "epsilon": epsilon,
                "n_samples": n_samples
            }
            
            # Save results and create visualizations
            try:
                result_file = results_path / f"results_epsilon_{epsilon}.json"
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=4)
                
                visualize_results(results, vis_path)
                
                logger.info(f"\nResults saved to {result_file}")
                logger.info(f"Visualizations saved to {vis_path}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in model operations: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.3,
                      help="Perturbation magnitude")
    parser.add_argument("--n_samples", type=int, default=100,
                      help="Number of test samples")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
    
    args = parser.parse_args()
    
    try:
        test_autoencoder_robustness(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)