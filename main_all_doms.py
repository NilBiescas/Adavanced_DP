import sys
import os
import yaml
import wandb
import torch

from src.Models import define_network
from src.train_loops_all_doms import baseline_train_all_domains

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YAML configuration
name_yaml = sys.argv[1]
with open(f"/fhome/amlai07/Adavanced_DP/Setups/{name_yaml}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize the model
model = define_network(config["model_params"]).to(device)

# Dataset selection
if ("dataset" not in config["dataset_params"]) or (config["dataset_params"]["dataset"] == "DomainNet"):
    from src.Loaders.DataLoaders import get_loaders
    train_loader, val_loader, test_loader = get_loaders(
        path=config["dataset_params"]["data_path"], 
        path_dn4il=config["dataset_params"].get("path_dn4il", ""),  # Adjust if needed
        image_size=config["dataset_params"]["image_size"], 
        batch_size=config["dataset_params"]["batch_size"], 
        config=config
    )
elif config["dataset_params"]["dataset"] == "DN4IL":
    from src.Loaders.DataLoaders_DN4IL_all_doms import get_loaders
    train_loader, val_loader, test_loader = get_loaders(
        path=config["dataset_params"]["data_path"], 
        path_dn4il=config["dataset_params"]["path_dn4il"], 
        image_size=config["dataset_params"]["image_size"], 
        batch_size=config["dataset_params"]["batch_size"], 
        config=config
    )
else:
    raise ValueError(f'{config["dataset_params"]["dataset"]} Dataset not supported')

# **Set the num_classes attribute on the model**
model.num_classes = train_loader.dataset.dataset.num_classes if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.num_classes

# import random
# # **Reduce dataset sizes to only 100 samples for train and validation loaders**
# def limit_dataset(dataloader, limit=100):
#     """
#     Randomly reduces the size of the dataloader to only `limit` samples.
#     Args:
#         - dataloader: The original PyTorch dataloader.
#         - limit (int): The number of samples to limit to.
#     Returns:
#         - A new dataloader with a limited dataset.
#     """
#     total_samples = len(dataloader.dataset)
#     indices = list(range(total_samples))
#     random.shuffle(indices)  # Shuffle the indices to make the sample random
#     indices = indices[:min(total_samples, limit)]  # Take a random subset
#     subset = torch.utils.data.Subset(dataloader.dataset, indices)
#     limited_loader = torch.utils.data.DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)
#     return limited_loader

# train_loader = limit_dataset(train_loader, limit=100)
# val_loader = limit_dataset(val_loader, limit=100)

# Initialize wandb
wandb.init(project="Advanced_DP", name=name_yaml)
wandb.config.update(config)

# Define optimizer
optimizer_config = {
    "lr": config["training_params"]["lr"], 
    "optimizer": config["training_params"]["optimizer"]
}
if optimizer_config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"])
elif optimizer_config["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["lr"])
elif optimizer_config["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config["lr"])
else:
    raise ValueError(f'{optimizer_config["optimizer"]} Optimizer not supported')

# Define loss function
if config["training_params"]["criterion"] == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
else:
    raise ValueError(f'{config["training_params"]["criterion"]} Criterion not supported')

# Scheduler configuration
scheduler_config = config["training_params"].get("scheduler", None)
if scheduler_config == "None":
    scheduler_config = None

# Set up directories
runs_dir = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}"
os.makedirs(runs_dir, exist_ok=True)
model.name = name_yaml

# Select training approach (all domains)
if ("Approach" not in config["training_params"]) or ("Baseline" == config["training_params"]["Approach"]):
    print("Baseline Approach (Training on All Domains)")
    model = baseline_train_all_domains(
        model=model, 
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        test_dataloader=test_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        epochs=config["training_params"]["epochs"], 
        early_stopping_patience=config["training_params"].get("early_stopping_patience", 5), 
        scheduler_config=scheduler_config, 
        alpha=config["training_params"].get("alpha", 1.0)
    )
else:
    raise ValueError(f'{config["training_params"]["Approach"]} Approach not supported')

# Save the final model
torch.save(model.state_dict(), f"{runs_dir}/model.pth")
wandb.finish()
