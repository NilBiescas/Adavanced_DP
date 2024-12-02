import sys
import os
import yaml
import wandb
import torch

from src.Models import define_network
from Adavanced_DP.src.train_loops import baseline_train, Task_Arithmetics_Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name_yaml = sys.argv[1]

with open(f"/fhome/amlai07/Adavanced_DP/Setups/{name_yaml}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = define_network(config["model_params"]).to(device)

print(config["dataset_params"]["dataset"])
if not "dataset" in config["dataset_params"] or config["dataset_params"]["dataset"] == "DomainNet": 
    from src.Loaders.DataLoaders import get_loaders
    train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])

elif config["dataset_params"]["dataset"] == "DN4IL":
    from src.Loaders.DataLoaders_DN4IL import get_loaders
    train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["path_dn4il"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])
else:
    raise ValueError(f'{config["dataset_params"]["dataset"]} Dataset not supported')


wandb.init(project="Advanced_DP", name=name_yaml)
wandb.config.update(config)


if config["training_params"]["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_params"]["lr"])
elif config["training_params"]["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training_params"]["lr"])
elif config["training_params"]["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["training_params"]["lr"])
else:
    raise ValueError(f'{config["training_params"]["optimizer"]} Optimizer not supported')

if config["training_params"]["criterion"] == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
else:
    raise ValueError(f'{config["training_params"]["criterion"]} Criterion not supported')

if "scheduler" not in config["training_params"] or config["training_params"]["scheduler"] == "None":
    config_scheduler = None
else:
    config_scheduler = config["training_params"]["scheduler"]

if "early_stopping_patience" not in config["training_params"]:
    config["training_params"]["early_stopping_patience"] = -1

# Save the model
if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs"):
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs")

if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}"):
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}")

model.name = name_yaml

if ("Approach" not in config["training_params"]) or ("Baseline" == config["training_params"]["Approach"]):
    print("Baseline Approach (Standard Finetuning)")
    model = baseline_train(model, train_loader, val_loader, test_loader, optimizer, criterion, device, config["training_params"]["epochs"], early_stopping_patience=config["training_params"]["early_stopping_patience"], scheduler_config=config_scheduler)
elif "TaskArithmetics" == config["training_params"]["Approach"]:
    print("Task Arithmetics Approach")
    optimizer_config = {"optimizer": config["training_params"]["optimizer"], "lr": config["training_params"]["lr"]}
    model = Task_Arithmetics_Train(model, train_loader, val_loader, test_loader, optimizer_config, criterion, device, config["training_params"]["epochs"], early_stopping_patience=config["training_params"]["early_stopping_patience"], scheduler_config=config_scheduler)
else:
    raise ValueError(f'{config["training_params"]["Approach"]} Approach not supported')

wandb.finish()


torch.save(model.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/model.pth")