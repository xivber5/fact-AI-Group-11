
import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Import saliency methods and models
from ViT_explanation_generator import Baselines
from ViT_new import vit_base_patch16_224
# from models.vgg import vgg19
import glob

from datasets.expl_hdf5 import ImagenetResults


# Normalize function
def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

# Function to calculate P-AUC
def calculate_p_auc(perturbation_steps, accuracies):
    # Approximate the area under the curve using the trapezoidal rule
    auc = np.trapz(accuracies, x=perturbation_steps)
    return auc

# Evaluation function
def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    perturbation_steps = np.arange(0, 0.33, 0.04)  # Attack radii (0/255 to 32/255 in steps of 4/255)
    p_auc_values = {method: [] for method in args.methods}

    # Load dataset
    vis_method_dir = os.path.join(args.visualizations_dir, args.method)
    imagenet_ds = ImagenetResults(vis_method_dir)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds, batch_size=args.batch_size, num_workers=2, shuffle=False
    )

    # Load model
    model = vit_base_patch16_224(pretrained=True).to(device)
    model.eval()

    # Iterate through all methods
    for method in args.methods:
        print(f"Evaluating method: {method}")
        accuracies = []

        for perturbation_step in perturbation_steps:
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (data, vis, target) in enumerate(tqdm(sample_loader)):
                data, vis, target = data.to(device), vis.to(device), target.to(device)

                # Normalize input
                norm_data = normalize(data.clone())
                pred = model(norm_data)
                pred_class = pred.argmax(dim=1)

                # Calculate initial accuracy
                if perturbation_step == 0:
                    total_samples += data.size(0)
                    correct_predictions += (pred_class == target).sum().item()

                # Apply perturbation
                vis_flat = vis.view(vis.size(0), -1)
                topk_pixels = int(224 * 224 * perturbation_step)
                _, topk_indices = torch.topk(vis_flat, k=topk_pixels, dim=-1)

                data_flat = data.view(data.size(0), -1)
                for i in range(data.size(0)):
                    data_flat[i, topk_indices[i]] = 0  # Zero-out top pixels

                perturbed_data = data_flat.view_as(data)
                norm_perturbed_data = normalize(perturbed_data)

                # Evaluate perturbed data
                perturbed_pred = model(norm_perturbed_data)
                perturbed_pred_class = perturbed_pred.argmax(dim=1)

                # Update accuracy
                correct_predictions += (perturbed_pred_class == target).sum().item()
                total_samples += data.size(0)

            accuracy = correct_predictions / total_samples
            accuracies.append(accuracy)

        # Calculate P-AUC for the method
        p_auc = calculate_p_auc(perturbation_steps, accuracies)
        p_auc_values[method] = p_auc

    # Save results
    np.save(os.path.join(args.output_dir, "p_auc_values.npy"), p_auc_values)

    # Plot results
    plot_results(perturbation_steps, p_auc_values, args.output_dir)


# Plotting function
def plot_results(perturbation_steps, p_auc_values, output_dir):
    plt.figure(figsize=(8, 5))
    
    for method, auc_values in p_auc_values.items():
        plt.plot(perturbation_steps, auc_values, label=method)
    
    plt.xlabel("Attack Radius (/255)")
    plt.ylabel("P-AUC")
    plt.legend()
    plt.grid(True)
    plt.title("P-AUC for Positive and Negative Perturbations")
    plt.savefig(os.path.join(output_dir, "p_auc_plot.png"))
    plt.show()


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturbation Evaluation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--method", type=str, default="transformer_attribution", help="Method to evaluate")
    parser.add_argument("--methods", type=list, default=["rollout", "gradcam", "full", "transformer_attribution"], help="Methods to compare")
    parser.add_argument("--visualizations-dir", type=str, required=True, help="Directory containing visualizations")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    eval(args)
