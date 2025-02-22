# import argparse
# import json
# import sys
# from datetime import datetime
# from pathlib import Path

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torcheval.metrics import MulticlassAccuracy

# from koopmann.data import DatasetConfig, MNISTDataset
# from koopmann.models import MLP, Autoencoder
# from koopmann.utils import compute_model_accuracy

# user = "gs4133"


# def fgsm_attack(model, loss_fn, data, epsilon, target):
#     """
#     Performs a single-step FGSM attack on the given data.
    
#     Args:
#         model (nn.Module): The trained model to attack.
#         loss_fn (callable): The loss function used for the model.
#         data (torch.Tensor): The input data to be perturbed.
#         epsilon (float): The magnitude of the perturbation.
#         target (torch.Tensor): The target labels for the input data.
    
#     Returns:
#         perturbed_data (torch.Tensor): The adversarially perturbed data.
#     """
#     # Make a copy so we don't modify the original data tensor in-place
#     adv_data = data.clone().detach().requires_grad_(True)

#     # Ensure gradients are zeroed out before the forward pass
#     model.zero_grad()
    
#     # Forward pass
#     outputs = model(adv_data)
    
#     # Compute the loss
#     loss = loss_fn(outputs, target)
    
#     # Backward pass to compute gradients
#     loss.backward()
    
#     # Collect the sign of the gradients
#     grad_sign = adv_data.grad.data.sign()
    
#     # Generate the perturbed data
#     perturbed_data = adv_data + epsilon * grad_sign
    
#     # Clamp the perturbed data to ensure valid pixel values in [-0.42, 2.3]
#     perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
#     return perturbed_data



# def cw_attack(model, data, target, c=1, kappa=0, steps=50, lr=0.01):

#     """

#     Implements C&W L2 attack.
    

#     Args:

#         model (nn.Module): Target model to attack
#         data (torch.Tensor): Input data
#         target (torch.Tensor): Target labels
#         c (float): Constant balancing loss terms
#         kappa (float): Confidence threshold
#         steps (int): Number of optimization steps
#         lr (float): Learning rate for optimization

    

#     Returns:
#         perturbed_data (torch.Tensor): Adversarially perturbed data
#     """

#     # Initialize variables
#     w = torch.zeros_like(data, requires_grad=True)
#     optimizer = torch.optim.Adam([w], lr=lr)
    

#     # Original data
#     data_orig = data.clone().detach()
#     target_onehot = F.one_hot(target, num_classes=10).float()

    

#     # Best results

#     best_adv = data.clone().detach()
#     best_l2 = 1e10 * torch.ones(data.shape[0]).to(data.device)

    

#     # CW attack loop

#     for step in range(steps):

#         # Forward pass with current perturbation

#         perturbed = torch.tanh(w) * 0.5 + 0.5
#         logits = model(perturbed)



#         # Calculate L2 distance

#         l2_dist = torch.sum((perturbed - data_orig) ** 2, dim=1)

        

#         # Calculate f(x + δ)

#         real = torch.sum(target_onehot * logits, 1)
#         other = torch.max((1 - target_onehot) * logits - (target_onehot * 10000), 1)[0]
#         f_loss = torch.clamp(other - real + kappa, min=0)

        

#         # Total loss

#         loss = l2_dist.sum() + c * f_loss.sum()

        

#         # Optimization step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

        

#         # Update best results

#         pred = logits.argmax(1)
#         is_adv = (pred != target)
#         is_better = l2_dist < best_l2
#         is_both = is_adv & is_better
#         best_l2[is_both] = l2_dist[is_both]
#         best_adv[is_both] = perturbed[is_both].clone().detach()

    

#     # Ensure output is in valid range
#     return torch.clamp(best_adv, -0.42, 2.3)






# def pgd_attack(model, loss_fn, data, epsilon, target, alpha=0.01, num_steps=40):
#     """
#     Performs a PGD attack on the given data.
    
#     Args:
#         model (nn.Module): The trained model to attack.
#         loss_fn (callable): The loss function used for the model.
#         data (torch.Tensor): The input data to be perturbed.
#         epsilon (float): Maximum perturbation magnitude.
#         target (torch.Tensor): Target labels.
#         alpha (float): Step size for each iteration (default: 0.01).
#         num_steps (int): Number of PGD iterations (default: 40).
    
#     Returns:
#         perturbed_data (torch.Tensor): The adversarially perturbed data.
#     """
#     # Initialize perturbed data with random noise
#     perturbed_data = data + torch.zeros_like(data).uniform_(-epsilon, epsilon)
#     perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
#     # Keep track of original data for projection
#     data_orig = data.clone().detach()
    
#     for step in range(num_steps):
#         # Make sure gradients are computed
#         perturbed_data.requires_grad = True
        
#         # Forward pass
#         outputs = model(perturbed_data)
        
#         # Calculate loss
#         loss = loss_fn(outputs, target)
        
#         # Zero all existing gradients
#         model.zero_grad()
        
#         # Calculate gradients
#         loss.backward()
        
#         # Create perturbation
#         with torch.no_grad():
#             # Take gradient step
#             perturbation = alpha * perturbed_data.grad.sign()
#             perturbed_data = perturbed_data + perturbation
            
#             # Project back to epsilon ball
#             delta = perturbed_data - data_orig
#             delta = torch.clamp(delta, -epsilon, epsilon)
#             perturbed_data = data_orig + delta
            
#             # Ensure valid pixel range
#             perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
#     return perturbed_data.detach()

# def mim_attack(model, loss_fn, data, epsilon, target, alpha=0.01, num_steps=40, decay_factor=1.0):
#     """
#     Momentum Iterative Method attack.
    
#     Args:
#         model (nn.Module): Target model to attack
#         loss_fn (callable): Loss function
#         data (torch.Tensor): Input data
#         epsilon (float): Maximum perturbation
#         target (torch.Tensor): Target labels
#         alpha (float): Step size
#         num_steps (int): Number of iterations
#         decay_factor (float): Momentum decay factor
#     """
#     # Initialize momentum and perturbed data
#     momentum = torch.zeros_like(data)
#     perturbed_data = data.clone().detach()
#     data_orig = data.clone().detach()
    
#     for step in range(num_steps):
#         # Compute gradient
#         perturbed_data.requires_grad = True
#         outputs = model(perturbed_data)
#         loss = loss_fn(outputs, target)
        
#         # Zero gradients
#         model.zero_grad()
#         loss.backward()
        
#         # Update momentum
#         grad = perturbed_data.grad.data
#         momentum = decay_factor * momentum + grad / torch.norm(grad, p=1)  # L1 normalization
        
#         # Update perturbed data
#         with torch.no_grad():
#             # Take step in direction of momentum
#             perturbed_data = perturbed_data + alpha * momentum.sign()
            
#             # Project back to epsilon ball
#             delta = perturbed_data - data_orig
#             delta = torch.clamp(delta, -epsilon, epsilon)
#             perturbed_data = data_orig + delta
            
#             # Ensure valid pixel range
#             perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
#     return perturbed_data.detach()

# def visualize_results(results, save_path):
#     """Visualize the results"""
#     plt.figure(figsize=(10, 6))
#     accuracies = [
#         results["original_accuracy"],
#         results["adversarial_accuracy"],
#         results["autoencoder_accuracy"],
#     ]
#     labels = ["Original", "Adversarial", "Autoencoder Protected"]
    
#     # Update title based on attack type
#     if results["attack_type"] in ['fgsm', 'pgd', 'mim']:
#         title_param = f"ε={results['epsilon']}"
#     elif results["attack_type"] == 'cw':
#         title_param = f"C={results['c_param']}"
    
#     sns.barplot(x=labels, y=accuracies)
#     plt.title(f"Accuracy Comparison ({title_param})")
#     plt.ylabel("Accuracy (%)")
    
#     # Save with appropriate filename
#     if results["attack_type"] in ['fgsm', 'pgd', 'mim']:
#         filename = f"accuracy_comparison_epsilon_{results['epsilon']}.png"
#     else:
#         filename = f"accuracy_comparison_c_{results['c_param']}.png"
    
#     plt.savefig(save_path / filename)
#     plt.close()


# class Generator(nn.Module):
#     """Generator network for AdvGAN"""
#     def __init__(self, input_dim=784):
#         super(Generator, self).__init__()
        
#         # Initial dense layer
#         self.fc = nn.Linear(input_dim, 1024)
        
#         # Main convolutional blocks
#         self.conv_blocks = nn.Sequential(
#             nn.Conv2d(1, 64, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
            
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#         )
        
#         # Transpose convolutions for upsampling
#         self.deconv_blocks = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
#             nn.Tanh()  # Output in [-1, 1]
#         )
        
#     def forward(self, x, epsilon=0.3):
#         # Reshape input if needed
#         if x.dim() == 2:
#             x = x.view(-1, 1, 28, 28)
            
#         # Generate perturbation
#         perturbation = self.deconv_blocks(self.conv_blocks(x))
        
#         # Scale perturbation by epsilon
#         perturbation = perturbation.view(x.shape) * epsilon
        
#         # Add perturbation to input
#         adv_x = torch.clamp(x + perturbation, -0.42, 2.3)
        
#         return adv_x

# def advgan_attack(model, generator, data, epsilon=0.3):
#     """
#     Performs AdvGAN attack using a pre-trained generator
    
#     Args:
#         model (nn.Module): Target model to attack
#         generator (nn.Module): Pre-trained AdvGAN generator
#         data (torch.Tensor): Input data to perturb
#         epsilon (float): Maximum perturbation magnitude
    
#     Returns:
#         perturbed_data (torch.Tensor): Adversarially perturbed data
#     """
#     generator.eval()
#     with torch.no_grad():
#         perturbed_data = generator(data, epsilon)
#     return perturbed_data

# def test_autoencoder_robustness(args):
#     data_root = Path(f"/scratch/{user}/data")
#     model_path = Path(f"/scratch/{user}/model_saves")

#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     n_samples = 60_000
#     epsilon = args.epsilon

#     dataset_config = DatasetConfig(
#         dataset_name="MNISTDataset",
#         num_samples=n_samples,
#         split="test",
#         torch_transform=None,
#         seed=42,
#     )

#     try:
#         test_dataset = MNISTDataset(
#             config=dataset_config,
#             root=str(data_root),
#             seed=42,
#         )
#         test_loader = torch.utils.data.DataLoader(
#             dataset=test_dataset,
#             batch_size=10000,
#             shuffle=False,
#         )
#         print(f"Successfully loaded dataset with {len(test_loader)} batches")

#     except Exception as e:
#         print(f"Error loading dataset: {str(e)}")
#         raise

#     # Load models
#     try:
#         # modify the file path
#         autoencoder_path = (
#             model_path / "k_1_dim_1024_loc_0_autoencoder_mnist_model.safetensors"
#         )
#         mlp_path = model_path / "mnist_probed.safetensors"

#         # check file path exist?
#         if not autoencoder_path.exists():
#             raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_path}")

#         if not mlp_path.exists():
#             raise FileNotFoundError(f"MLP model not found: {mlp_path}")

#         print(f"Loading autoencoder from: {autoencoder_path}")
#         autoencoder, _ = Autoencoder.load_model(autoencoder_path)
#         autoencoder = autoencoder.to(device)
#         autoencoder.eval()

#         print(f"Loading MLP from: {mlp_path}")
#         mlp, _ = MLP.load_model(mlp_path)

#         # NOTE: Remove nonlinearities from the probe!
#         mlp.modules[-2].remove_nonlinearity()
#         mlp.modules[-3].remove_nonlinearity()

#         mlp.to(device).eval()

#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         raise

#     # Create and reset metrics
#     mlp_original_metric = MulticlassAccuracy().to(device)
#     mlp_adversarial_metric = MulticlassAccuracy().to(device)
#     koopman_aug_metric = MulticlassAccuracy().to(device)

#     # Reset metrics explicitly
#     mlp_original_metric.reset()
#     mlp_adversarial_metric.reset()
#     koopman_aug_metric.reset()

#     print(f"\nRunning {args.attack.upper()} attack")
    
#     for i, (data, target) in enumerate(test_loader):
#         if i >= 1:
#             break

#         data, target = data.to(device), target.to(device)
#         data = data.flatten(start_dim=1)

#         # Load or create AdvGAN generator if needed
#         if args.attack == 'advgan':
#             generator = Generator().to(device)
#             generator_path = model_path / "advgan_generator.safetensors"
#             if generator_path.exists():
#                 generator.load_state_dict(torch.load(generator_path))
#             else:
#                 print("Warning: Pre-trained generator not found, using untrained generator")
        
#         # Choose attack based on args.attack
#         if args.attack == 'fgsm':
#             perturbed_data = fgsm_attack(mlp, nn.CrossEntropyLoss(), data, args.epsilon, target)
#         elif args.attack == 'pgd':
#             perturbed_data = pgd_attack(
#                 mlp, nn.CrossEntropyLoss(), data, args.epsilon, target,
#                 alpha=args.alpha, num_steps=args.num_steps
#             )
#         elif args.attack == 'mim':
#             perturbed_data = mim_attack(
#                 mlp, nn.CrossEntropyLoss(), data, args.epsilon, target,
#                 alpha=args.alpha, num_steps=args.num_steps,
#                 decay_factor=args.decay_factor
#             )
#         elif args.attack == 'cw':
#             perturbed_data = cw_attack(
#                 mlp, data, target,
#                 c=args.c, kappa=args.kappa, steps=args.cw_steps, lr=args.cw_lr
#             )
#         elif args.attack == 'advgan':
#             perturbed_data = advgan_attack(mlp, generator, data, args.epsilon)

#         with torch.no_grad():
#             # Test MLP with original data
#             original_output = mlp(data)
#             mlp_original_metric.update(original_output, target.squeeze())

#             # Test MLP with adversarial data
#             adv_output = mlp(perturbed_data)
#             mlp_adversarial_metric.update(adv_output, target.squeeze())

#             # Test with autoencoder's latent representation
#             predict_act = autoencoder(perturbed_data, k=1).predictions[-1]
#             koopman_aug_output = mlp.modules[-2:](predict_act)
#             koopman_aug_metric.update(koopman_aug_output, target.squeeze())

#             # Print perturbation statistics
#             perturbation = perturbed_data - data
#             l2_dist = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1).mean()
#             max_diff = torch.max(torch.abs(perturbation))
#             print(f"\nPerturbation statistics:")
#             print(f"  L2 distance: {l2_dist:.4f}")
#             print(f"  Max difference: {max_diff:.4f}")

#     mlp_original_acc = mlp_original_metric.compute()
#     mlp_adv_acc = mlp_adversarial_metric.compute()
#     koopman_acc = koopman_aug_metric.compute()

#     print(f"Original accuracy for a single batch: {mlp_original_acc}")
#     print(f"Adversarial attack accuracy for a single batch: {mlp_adv_acc}")
#     print(f"Koopman accuracy for a single batch: {koopman_acc}")

#     # Update results to include attack type and appropriate parameters
#     results = {
#         "attack_type": args.attack,
#         "original_accuracy": mlp_original_acc.item(),
#         "adversarial_accuracy": mlp_adv_acc.item(),
#         "koopman_accuracy": koopman_acc.item(),
#     }

#     # Add attack-specific parameters
#     if args.attack in ['fgsm', 'pgd', 'mim']:
#         results["epsilon"] = args.epsilon
#     elif args.attack == 'cw':
#         results["c_param"] = args.c
#         results["kappa"] = args.kappa

#     # Print results with appropriate parameters
#     print(f"\nResults for {args.attack.upper()} attack:")
#     if args.attack in ['fgsm', 'pgd', 'mim']:
#         print(f"  Epsilon: {args.epsilon}")
#     elif args.attack == 'cw':
#         print(f"  C: {args.c}, Kappa: {args.kappa}")
#     print(f"  Original accuracy: {mlp_original_acc:.4f}")
#     print(f"  Adversarial accuracy: {mlp_adv_acc:.4f}")
#     print(f"  Koopman accuracy: {koopman_acc:.4f}")

#     return {
#         'original_acc': mlp_original_acc,
#         'adversarial_acc': mlp_adv_acc,
#         'koopman_acc': koopman_acc,
#         'attack_params': args.epsilon if args.attack in ['fgsm', 'pgd', 'mim'] else args.c
#     }


# def compare_attacks(model_path):
#     """Run and compare different attacks"""
    
#     # Define attack configurations
#     attacks_config = {
#         'fgsm': {'epsilon': [0.1, 0.2, 0.3]},
#         'pgd': {
#             'epsilon': [0.1, 0.2, 0.3],
#             'alpha': 0.01,
#             'num_steps': 40
#         },
#         'mim': {
#             'epsilon': [0.1, 0.2, 0.3],
#             'alpha': 0.01,
#             'num_steps': 40,
#             'decay_factor': 1.0
#         },
#         'cw': {
#             'c': [0.0, 0.1, 1.0, 10.0],  # Added c=0
#             'kappa': [0.0, 1000.0],  # Added k=1000
#             'steps': 50,
#             'lr': 0.01
#         }
#     }

#     results_list = []
    
#     for attack_type, config in attacks_config.items():
#         print(f"\nTesting {attack_type.upper()} attack...")
        
#         args = argparse.Namespace()
#         args.attack = attack_type
        
#         # Initialize all possible parameters with defaults
#         args.epsilon = None
#         args.c = None
#         args.kappa = 0.0
#         args.cw_steps = 50
#         args.cw_lr = 0.01
#         args.alpha = 0.01
#         args.num_steps = 40
#         args.decay_factor = 1.0
        
#         # Set specific parameters based on attack type
#         if attack_type == 'cw':
#             param_name = 'C'
#             c_values = config['c']
#             kappa_values = config['kappa']
#             args.cw_steps = config['steps']
#             args.cw_lr = config['lr']
            
#             # Test all combinations of c and kappa
#             for c in c_values:
#                 for k in kappa_values:
#                     args.c = c
#                     args.kappa = k
                    
#                     # Run attack
#                     results = test_autoencoder_robustness(args)
                    
#                     # Store results
#                     results_list.append({
#                         'Attack': 'CW',
#                         'C': c,
#                         'Kappa': k,
#                         'Original Acc': results['original_acc'].item(),
#                         'Adversarial Acc': results['adversarial_acc'].item(),
#                         'Koopman Acc': results['koopman_acc'].item()
#                     })
#         else:
#             param_name = 'Epsilon'
#             param_values = config['epsilon']
#             if attack_type in ['pgd', 'mim']:
#                 args.alpha = config['alpha']
#                 args.num_steps = config['num_steps']
#             if attack_type == 'mim':
#                 args.decay_factor = config['decay_factor']
            
#             for value in param_values:
#                 args.epsilon = value
                
#                 # Run attack
#                 results = test_autoencoder_robustness(args)
                
#                 # Store results
#                 results_list.append({
#                     'Attack': attack_type.upper(),
#                     param_name: value,
#                     'Original Acc': results['original_acc'].item(),
#                     'Adversarial Acc': results['adversarial_acc'].item(),
#                     'Koopman Acc': results['koopman_acc'].item()
#                 })
    
#     # Create DataFrame
#     results_df = pd.DataFrame(results_list)
    
#     # Save results
#     csv_path = model_path / 'attack_comparison.csv'
#     results_df.to_csv(csv_path, index=False)
#     print(f"\nResults saved to: {csv_path}")
    
#     # Create styled HTML
#     styled_df = results_df.style.format({
#         'Original Acc': '{:.4f}',
#         'Adversarial Acc': '{:.4f}',
#         'Koopman Acc': '{:.4f}'
#     })
#     html_path = model_path / 'attack_comparison.html'
#     styled_df.to_html(html_path)
    
#     # Display results
#     print("\nAttack Comparison Results:")
#     print(results_df.to_string(index=False))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     # Add compare flag before other arguments
#     parser.add_argument("--compare", action="store_true", 
#                        help="Run comparison of all attacks")
    
#     # Attack type and parameters
#     parser.add_argument("--attack", type=str, 
#                        choices=['fgsm', 'pgd', 'mim', 'cw'], 
#                        default='fgsm', 
#                        required=False,  # Not required when using --compare
#                        help="Attack type")
    
#     # Attack parameters
#     parser.add_argument("--epsilon", type=float, default=0.2, 
#                        help="Perturbation magnitude for FGSM/PGD/MIM")
#     parser.add_argument("--alpha", type=float, default=0.01, 
#                        help="Step size for PGD/MIM")
#     parser.add_argument("--num_steps", type=int, default=40, 
#                        help="Number of iterations for PGD/MIM")
#     parser.add_argument("--decay_factor", type=float, default=1.0, 
#                        help="Momentum decay factor for MIM")
    
#     # CW attack parameters
#     parser.add_argument("--c", type=float, default=1.0, 
#                        help="CW confidence parameter")
#     parser.add_argument("--kappa", type=float, default=0.0, 
#                        help="CW confidence threshold")
#     parser.add_argument("--cw_steps", type=int, default=50, 
#                        help="Number of CW optimization steps")
#     parser.add_argument("--cw_lr", type=float, default=0.01, 
#                        help="CW learning rate")

#     args = parser.parse_args()

#     try:
#         if args.compare:
#             print("\nRunning comparison of all attacks...")
#             model_path = Path(f"/scratch/{user}/model_saves")
#             compare_attacks(model_path)
#         else:
#             test_autoencoder_robustness(args)
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)



import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
import foolbox as fb

from koopmann.data import DatasetConfig, MNISTDataset
from koopmann.models import MLP, Autoencoder
from koopmann.utils import compute_model_accuracy

user = "gs4133"


def fgsm_attack(model, loss_fn, data, epsilon, target):
    """
    Performs a single-step FGSM attack on the given data.
    
    Args:
        model (nn.Module): The trained model to attack.
        loss_fn (callable): The loss function used for the model.
        data (torch.Tensor): The input data to be perturbed.
        epsilon (float): The magnitude of the perturbation.
        target (torch.Tensor): The target labels for the input data.
    
    Returns:
        perturbed_data (torch.Tensor): The adversarially perturbed data.
    """
    # Make a copy so we don't modify the original data tensor in-place
    adv_data = data.clone().detach().requires_grad_(True)

    # Ensure gradients are zeroed out before the forward pass
    model.zero_grad()
    
    # Forward pass
    outputs = model(adv_data)
    
    # Compute the loss
    loss = loss_fn(outputs, target)
    
    # Backward pass to compute gradients
    loss.backward()
    
    # Collect the sign of the gradients
    grad_sign = adv_data.grad.data.sign()
    
    # Generate the perturbed data
    perturbed_data = adv_data + epsilon * grad_sign
    
    # Clamp the perturbed data to ensure valid pixel values in [-0.42, 2.3]
    perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
    return perturbed_data

def cw_attack(model, data, target, c=1, kappa=0, steps=50, lr=0.01, binary_search_steps=5):
    """
    Implements C&W L2 attack using Foolbox.
    
    Args:
        model (nn.Module): Target model to attack
        data (torch.Tensor): Input data
        target (torch.Tensor): Target labels
        c (float): Trade-off constant
        kappa (float): Confidence threshold
        steps (int): Number of optimization steps
        lr (float): Learning rate
        binary_search_steps (int): Number of binary search steps
    """
    # Wrap the PyTorch model for Foolbox
    preprocessing = dict(mean=[0.0], std=[2.3+0.42])  # No preprocessing needed for MNIST
    bounds = (-0.42, 2.3)  # Valid pixel range
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    
    # Initialize the CW L2 attack
    attack = fb.attacks.L2CarliniWagnerAttack(
        binary_search_steps=binary_search_steps,
        steps=steps,
        stepsize=lr,
        confidence=kappa,
        initial_const=c,
        abort_early=True
    )
    
    # Generate adversarial examples
    _, perturbed_data, success = attack(fmodel, data, target, epsilons=None)
    
    # Print attack success rate
    success_rate = success.float().mean().item()
    print(f"\nCW Attack Success Rate: {success_rate:.4f}")
    
    return perturbed_data






def pgd_attack(model, loss_fn, data, epsilon, target, alpha=0.01, num_steps=40):
    """
    Performs a PGD attack on the given data.
    
    Args:
        model (nn.Module): The trained model to attack.
        loss_fn (callable): The loss function used for the model.
        data (torch.Tensor): The input data to be perturbed.
        epsilon (float): Maximum perturbation magnitude.
        target (torch.Tensor): Target labels.
        alpha (float): Step size for each iteration (default: 0.01).
        num_steps (int): Number of PGD iterations (default: 40).
    
    Returns:
        perturbed_data (torch.Tensor): The adversarially perturbed data.
    """
    # Initialize perturbed data with random noise
    perturbed_data = data + torch.zeros_like(data).uniform_(-epsilon, epsilon)
    perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
    # Keep track of original data for projection
    data_orig = data.clone().detach()
    
    for step in range(num_steps):
        # Make sure gradients are computed
        perturbed_data.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_data)
        
        # Calculate loss
        loss = loss_fn(outputs, target)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Create perturbation
        with torch.no_grad():
            # Take gradient step
            perturbation = alpha * perturbed_data.grad.sign()
            perturbed_data = perturbed_data + perturbation
            
            # Project back to epsilon ball
            delta = perturbed_data - data_orig
            delta = torch.clamp(delta, -epsilon, epsilon)
            perturbed_data = data_orig + delta
            
            # Ensure valid pixel range
            perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
    return perturbed_data.detach()

def mim_attack(model, loss_fn, data, epsilon, target, alpha=0.01, num_steps=40, decay_factor=1.0):
    """
    Momentum Iterative Method attack.
    
    Args:
        model (nn.Module): Target model to attack
        loss_fn (callable): Loss function
        data (torch.Tensor): Input data
        epsilon (float): Maximum perturbation
        target (torch.Tensor): Target labels
        alpha (float): Step size
        num_steps (int): Number of iterations
        decay_factor (float): Momentum decay factor
    """
    # Initialize momentum and perturbed data
    momentum = torch.zeros_like(data)
    perturbed_data = data.clone().detach()
    data_orig = data.clone().detach()
    
    for step in range(num_steps):
        # Compute gradient
        perturbed_data.requires_grad = True
        outputs = model(perturbed_data)
        loss = loss_fn(outputs, target)
        
        # Zero gradients
        model.zero_grad()
        loss.backward()
        
        # Update momentum
        grad = perturbed_data.grad.data
        momentum = decay_factor * momentum + grad / torch.norm(grad, p=1)  # L1 normalization
        
        # Update perturbed data
        with torch.no_grad():
            # Take step in direction of momentum
            perturbed_data = perturbed_data + alpha * momentum.sign()
            
            # Project back to epsilon ball
            delta = perturbed_data - data_orig
            delta = torch.clamp(delta, -epsilon, epsilon)
            perturbed_data = data_orig + delta
            
            # Ensure valid pixel range
            perturbed_data = torch.clamp(perturbed_data, -0.42, 2.3)
    
    return perturbed_data.detach()

def visualize_results(results, save_path):
    """Visualize the results"""
    plt.figure(figsize=(10, 6))
    accuracies = [
        results["original_accuracy"],
        results["adversarial_accuracy"],
        results["autoencoder_accuracy"],
    ]
    labels = ["Original", "Adversarial", "Autoencoder Protected"]
    
    # Update title based on attack type
    if results["attack_type"] in ['fgsm', 'pgd', 'mim']:
        title_param = f"ε={results['epsilon']}"
    elif results["attack_type"] == 'cw':
        title_param = f"C={results['c_param']}"
    
    sns.barplot(x=labels, y=accuracies)
    plt.title(f"Accuracy Comparison ({title_param})")
    plt.ylabel("Accuracy (%)")
    
    # Save with appropriate filename
    if results["attack_type"] in ['fgsm', 'pgd', 'mim']:
        filename = f"accuracy_comparison_epsilon_{results['epsilon']}.png"
    else:
        filename = f"accuracy_comparison_c_{results['c_param']}.png"
    
    plt.savefig(save_path / filename)
    plt.close()


class Generator(nn.Module):
    """Generator network for AdvGAN"""
    def __init__(self, input_dim=784):
        super(Generator, self).__init__()
        
        # Initial dense layer
        self.fc = nn.Linear(input_dim, 1024)
        
        # Main convolutional blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Transpose convolutions for upsampling
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, x, epsilon=0.3):
        # Reshape input if needed
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
            
        # Generate perturbation
        perturbation = self.deconv_blocks(self.conv_blocks(x))
        
        # Scale perturbation by epsilon
        perturbation = perturbation.view(x.shape) * epsilon
        
        # Add perturbation to input
        adv_x = torch.clamp(x + perturbation, -0.42, 2.3)
        
        return adv_x

def advgan_attack(model, generator, data, epsilon=0.3):
    """
    Performs AdvGAN attack using a pre-trained generator
    
    Args:
        model (nn.Module): Target model to attack
        generator (nn.Module): Pre-trained AdvGAN generator
        data (torch.Tensor): Input data to perturb
        epsilon (float): Maximum perturbation magnitude
    
    Returns:
        perturbed_data (torch.Tensor): Adversarially perturbed data
    """
    generator.eval()
    with torch.no_grad():
        perturbed_data = generator(data, epsilon)
    return perturbed_data

def test_autoencoder_robustness(args):
    data_root = Path(f"/scratch/{user}/data")
    model_path = Path(f"/scratch/{user}/model_saves")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_samples = 60_000
    epsilon = args.epsilon

    dataset_config = DatasetConfig(
        dataset_name="MNISTDataset",
        num_samples=n_samples,
        split="test",
        torch_transform=None,
        seed=42,
    )

    try:
        test_dataset = MNISTDataset(
            config=dataset_config,
            root=str(data_root),
            seed=42,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=10000,
            shuffle=False,
        )
        print(f"Successfully loaded dataset with {len(test_loader)} batches")

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

    # Load models
    try:
        # modify the file path
        autoencoder_path = (
            model_path / "k_1_dim_1024_loc_0_autoencoder_mnist_model.safetensors"
        )
        mlp_path = model_path / "mnist_probed.safetensors"

        # check file path exist?
        if not autoencoder_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_path}")

        if not mlp_path.exists():
            raise FileNotFoundError(f"MLP model not found: {mlp_path}")

        print(f"Loading autoencoder from: {autoencoder_path}")
        autoencoder, _ = Autoencoder.load_model(autoencoder_path)
        autoencoder = autoencoder.to(device)
        autoencoder.eval()

        print(f"Loading MLP from: {mlp_path}")
        mlp, _ = MLP.load_model(mlp_path)

        # NOTE: Remove nonlinearities from the probe!
        mlp.modules[-2].remove_nonlinearity()
        mlp.modules[-3].remove_nonlinearity()

        mlp.to(device).eval()

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    # Create and reset metrics
    mlp_original_metric = MulticlassAccuracy().to(device)
    mlp_adversarial_metric = MulticlassAccuracy().to(device)
    koopman_aug_metric = MulticlassAccuracy().to(device)

    # Reset metrics explicitly
    mlp_original_metric.reset()
    mlp_adversarial_metric.reset()
    koopman_aug_metric.reset()

    print(f"\nRunning {args.attack.upper()} attack")
    
    for i, (data, target) in enumerate(test_loader):
        if i >= 1:
            break

        data, target = data.to(device), target.to(device)
        data = data.flatten(start_dim=1)

        # Load or create AdvGAN generator if needed
        if args.attack == 'advgan':
            generator = Generator().to(device)
            generator_path = model_path / "advgan_generator.safetensors"
            if generator_path.exists():
                generator.load_state_dict(torch.load(generator_path))
            else:
                print("Warning: Pre-trained generator not found, using untrained generator")
        
        # Choose attack based on args.attack
        if args.attack == 'fgsm':
            perturbed_data = fgsm_attack(mlp, nn.CrossEntropyLoss(), data, args.epsilon, target)
        elif args.attack == 'pgd':
            perturbed_data = pgd_attack(
                mlp, nn.CrossEntropyLoss(), data, args.epsilon, target,
                alpha=args.alpha, num_steps=args.num_steps
            )
        elif args.attack == 'mim':
            perturbed_data = mim_attack(
                mlp, nn.CrossEntropyLoss(), data, args.epsilon, target,
                alpha=args.alpha, num_steps=args.num_steps,
                decay_factor=args.decay_factor
            )
        elif args.attack == 'cw':
            # Reshape data for Foolbox (N, C, H, W)
            data_reshaped = data.view(-1, 1, 28, 28)
            perturbed_data = cw_attack(
                mlp, 
                data_reshaped, 
                target,
                c=args.c, 
                kappa=args.kappa, 
                steps=args.cw_steps, 
                lr=args.cw_lr,
                binary_search_steps=args.binary_search_steps
            )
            # Reshape back to flattened form
            perturbed_data = perturbed_data.view(data.shape)
        elif args.attack == 'advgan':
            perturbed_data = advgan_attack(mlp, generator, data, args.epsilon)

        with torch.no_grad():
            # Test MLP with original data
            original_output = mlp(data)
            mlp_original_metric.update(original_output, target.squeeze())

            # Test MLP with adversarial data
            adv_output = mlp(perturbed_data)
            mlp_adversarial_metric.update(adv_output, target.squeeze())

            # Test with autoencoder's latent representation
            predict_act = autoencoder(perturbed_data, k=1).predictions[-1]
            koopman_aug_output = mlp.modules[-2:](predict_act)
            koopman_aug_metric.update(koopman_aug_output, target.squeeze())

            # Print perturbation statistics
            perturbation = perturbed_data - data
            l2_dist = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1).mean()
            max_diff = torch.max(torch.abs(perturbation))
            print(f"\nPerturbation statistics:")
            print(f"  L2 distance: {l2_dist:.4f}")
            print(f"  Max difference: {max_diff:.4f}")

    mlp_original_acc = mlp_original_metric.compute()
    mlp_adv_acc = mlp_adversarial_metric.compute()
    koopman_acc = koopman_aug_metric.compute()

    print(f"Original accuracy for a single batch: {mlp_original_acc}")
    print(f"Adversarial attack accuracy for a single batch: {mlp_adv_acc}")
    print(f"Koopman accuracy for a single batch: {koopman_acc}")

    # Update results to include attack type and appropriate parameters
    results = {
        "attack_type": args.attack,
        "original_accuracy": mlp_original_acc.item(),
        "adversarial_accuracy": mlp_adv_acc.item(),
        "koopman_accuracy": koopman_acc.item(),
    }

    # Add attack-specific parameters
    if args.attack in ['fgsm', 'pgd', 'mim']:
        results["epsilon"] = args.epsilon
    elif args.attack == 'cw':
        results["c_param"] = args.c
        results["kappa"] = args.kappa

    # Print results with appropriate parameters
    print(f"\nResults for {args.attack.upper()} attack:")
    if args.attack in ['fgsm', 'pgd', 'mim']:
        print(f"  Epsilon: {args.epsilon}")
    elif args.attack == 'cw':
        print(f"  C: {args.c}, Kappa: {args.kappa}")
    print(f"  Original accuracy: {mlp_original_acc:.4f}")
    print(f"  Adversarial accuracy: {mlp_adv_acc:.4f}")
    print(f"  Koopman accuracy: {koopman_acc:.4f}")

    return {
        'original_acc': mlp_original_acc,
        'adversarial_acc': mlp_adv_acc,
        'koopman_acc': koopman_acc,
        'attack_params': args.epsilon if args.attack in ['fgsm', 'pgd', 'mim'] else args.c
    }


def compare_attacks(model_path):
    """Run and compare different attacks"""
    
    # Define attack configurations
    attacks_config = {
        'fgsm': {'epsilon': [0.1, 0.2, 0.3]},
        'pgd': {
            'epsilon': [0.1, 0.2, 0.3],
            'alpha': 0.01,
            'num_steps': 40
        },
        'mim': {
            'epsilon': [0.1, 0.2, 0.3],
            'alpha': 0.01,
            'num_steps': 40,
            'decay_factor': 1.0
        },
        'cw': {
            'c': [0.0, 0.1, 1.0, 10.0],  # Added c=0
            'kappa': [0.0, 1000.0],  # Added k=1000
            'steps': 50,
            'lr': 0.01
        }
    }

    results_list = []
    
    for attack_type, config in attacks_config.items():
        print(f"\nTesting {attack_type.upper()} attack...")
        
        args = argparse.Namespace()
        args.attack = attack_type
        
        # Initialize all possible parameters with defaults
        args.epsilon = None
        args.c = None
        args.kappa = 0.0
        args.cw_steps = 50
        args.cw_lr = 0.01
        args.alpha = 0.01
        args.num_steps = 40
        args.decay_factor = 1.0
        
        # Set specific parameters based on attack type
        if attack_type == 'cw':
            param_name = 'C'
            c_values = config['c']
            kappa_values = config['kappa']
            args.cw_steps = config['steps']
            args.cw_lr = config['lr']
            
            # Test all combinations of c and kappa
            for c in c_values:
                for k in kappa_values:
                    args.c = c
                    args.kappa = k
                    
                    # Run attack
                    results = test_autoencoder_robustness(args)
                    
                    # Store results
                    results_list.append({
                        'Attack': 'CW',
                        'C': c,
                        'Kappa': k,
                        'Original Acc': results['original_acc'].item(),
                        'Adversarial Acc': results['adversarial_acc'].item(),
                        'Koopman Acc': results['koopman_acc'].item()
                    })
        else:
            param_name = 'Epsilon'
            param_values = config['epsilon']
            if attack_type in ['pgd', 'mim']:
                args.alpha = config['alpha']
                args.num_steps = config['num_steps']
            if attack_type == 'mim':
                args.decay_factor = config['decay_factor']
            
            for value in param_values:
                args.epsilon = value
                
                # Run attack
                results = test_autoencoder_robustness(args)
                
                # Store results
                results_list.append({
                    'Attack': attack_type.upper(),
                    param_name: value,
                    'Original Acc': results['original_acc'].item(),
                    'Adversarial Acc': results['adversarial_acc'].item(),
                    'Koopman Acc': results['koopman_acc'].item()
                })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results
    csv_path = model_path / 'attack_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create styled HTML
    styled_df = results_df.style.format({
        'Original Acc': '{:.4f}',
        'Adversarial Acc': '{:.4f}',
        'Koopman Acc': '{:.4f}'
    })
    html_path = model_path / 'attack_comparison.html'
    styled_df.to_html(html_path)
    
    # Display results
    print("\nAttack Comparison Results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add compare flag before other arguments
    parser.add_argument("--compare", action="store_true", 
                       help="Run comparison of all attacks")
    
    # Attack type and parameters
    parser.add_argument("--attack", type=str, 
                       choices=['fgsm', 'pgd', 'mim', 'cw'], 
                       default='fgsm', 
                       required=False,  # Not required when using --compare
                       help="Attack type")
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=0.2, 
                       help="Perturbation magnitude for FGSM/PGD/MIM")
    parser.add_argument("--alpha", type=float, default=0.01, 
                       help="Step size for PGD/MIM")
    parser.add_argument("--num_steps", type=int, default=40, 
                       help="Number of iterations for PGD/MIM")
    parser.add_argument("--decay_factor", type=float, default=1.0, 
                       help="Momentum decay factor for MIM")
    
    # CW attack parameters
    parser.add_argument("--c", type=float, default=1.0, 
                       help="Initial CW trade-off constant")
    parser.add_argument("--kappa", type=float, default=0.0, 
                       help="CW confidence threshold")
    parser.add_argument("--cw_steps", type=int, default=1000, 
                       help="Number of CW optimization steps")
    parser.add_argument("--cw_lr", type=float, default=0.01, 
                       help="CW learning rate")
    parser.add_argument("--binary_search_steps", type=int, default=9,
                       help="Number of binary search steps for CW")

    args = parser.parse_args()

    try:
        if args.compare:
            print("\nRunning comparison of all attacks...")
            model_path = Path(f"/scratch/{user}/model_saves")
            compare_attacks(model_path)
        else:
            test_autoencoder_robustness(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
