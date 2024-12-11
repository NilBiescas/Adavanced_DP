import torch
from tqdm import tqdm
import os
from src.Models import define_network
from src.Loaders.DataLoaders_DN4IL_all_doms import get_loaders
from src.train_loops_all_doms import compute_per_domain_accuracy
from src.utils.plotting_functions import plot_strictly_lower_triangular_heatmap

def compute_and_plot_top5_heatmap(model_path, dataloader, device, num_domains, idx2domain, save_path):

    # Initialize the model architecture
    config = {
        "model_params": {
            "type": "baseline",  # Adjust to match your model type
            "num_classes": 100  # Adjust to the number of classes in the dataset
        }
    }
    model = define_network(config["model_params"]).to(device)

    # Set num_classes directly during model definition
    if not hasattr(model, 'num_classes'):
        model.num_classes = config["model_params"]["num_classes"]

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Compute per-domain accuracy
    _, per_domain_top5_acc = compute_per_domain_accuracy(model, dataloader, device, num_domains)

    # Plot the heatmap
    plot_strictly_lower_triangular_heatmap(
        data=[per_domain_top5_acc],
        labels=[idx2domain[i] for i in range(num_domains)],
        save_path=save_path
    )
    print(f"Heatmap saved at {save_path}")

if __name__ == "__main__":
    # Configuration
    model_path = "/fhome/amlai07/Adavanced_DP/Runs/baseline_all_doms/best_model_all_domains.pth"  # Adjust to your saved model path
    save_path = "/fhome/amlai07/Adavanced_DP/Runs/baseline_all_doms/results/heatmap_top5.png"  # Adjust to desired heatmap save location
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the data
    dataset_config = {
        "dataset_params": {
            "data_path": "/fhome/amlai07/Adavanced_DP/Data/domainnet",
            "path_dn4il": "/fhome/amlai07/Adavanced_DP/Data/DN4IL",
            "image_size": 384,
            "batch_size": 25,
            "num_domains": 6
        }
    }
    _, _, test_loader = get_loaders(
        path=dataset_config["dataset_params"]["data_path"],
        path_dn4il=dataset_config["dataset_params"]["path_dn4il"],
        image_size=dataset_config["dataset_params"]["image_size"],
        batch_size=dataset_config["dataset_params"]["batch_size"],
        config=dataset_config
    )

    # Get domain info
    num_domains = test_loader.dataset.num_domains
    idx2domain = {i: domain for i, domain in enumerate(test_loader.dataset.domains)}

    # Compute and plot
    compute_and_plot_top5_heatmap(model_path, test_loader, device, num_domains, idx2domain, save_path)
