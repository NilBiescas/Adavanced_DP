import sys
import os
import yaml
import wandb
import torch

from src.Models import define_network
from src.train_loops_all_doms import baseline_train_all_domains

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name_yaml = sys.argv[1]
with open(f"/fhome/amlai07/Adavanced_DP/Setups/{name_yaml}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = define_network(config["model_params"]).to(device)

if ("dataset" not in config["dataset_params"]) or (config["dataset_params"]["dataset"] == "DomainNet"):
    from src.Loaders.DataLoaders import get_loaders
    train_loader, val_loader, test_loader = get_loaders(
        path=config["dataset_params"]["data_path"], 
        path_dn4il=config["dataset_params"].get("path_dn4il", ""),
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

model.num_classes = train_loader.dataset.dataset.num_classes if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.num_classes

wandb.init(project="Advanced_DP", name=name_yaml)
wandb.config.update(config)

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

if config["training_params"]["criterion"] == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
else:
    raise ValueError(f'{config["training_params"]["criterion"]} Criterion not supported')

scheduler_config = config["training_params"].get("scheduler", None)
if scheduler_config == "None":
    scheduler_config = None

runs_dir = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}"
os.makedirs(runs_dir, exist_ok=True)
model.name = name_yaml

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

torch.save(model.state_dict(), f"{runs_dir}/model.pth")
wandb.finish()
