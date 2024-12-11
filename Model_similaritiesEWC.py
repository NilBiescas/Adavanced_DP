import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import sys
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.task_vectors import TaskVector
from src.utils.evaluateFunctions_and_definiOptimizer import top1_and_top_k_accuracy_domain
from src.utils.plotting_functions import plot_strictly_lower_triangular_heatmap, createCSV, plotStablityPlasticity, generate_plot_practica
from src.utils.ewc_functions import add_importances
from src.Models import define_network


def compute_loss(state_dict, old_model_state_dict, importances):
    regularitzation_loss = 0
    for name, param in state_dict.items():
        if (name in old_model_state_dict) and (name in importances):
            regularitzation_loss += (importances[name] * (param - old_model_state_dict[name]).pow(2)).sum() 

    return regularitzation_loss

def compute_importances(model, data_loader, criterion, device):
    model.eval()
    model.zero_grad()
    importances = {}
    for name, param in model.named_parameters():
        importances[name] = torch.zeros_like(param)

    for inputs, targets in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad.data is not None:
                importances[name] += param.grad.data ** 2
        model.zero_grad()

    for name, param in model.named_parameters():
        importances[name] /= len(data_loader)

    return importances


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    name_yaml_EWC = "TeacherStudent_dn4il_EWC_student_only_v4_Dn4il_domain_order"
    name_yaml_NO_EWC = "TeacherStudent_dn4il_domain_order"
    config = yaml.load(open(f"Setups/{name_yaml_EWC}.yaml", "r"), Loader=yaml.FullLoader)

    name_yaml_plots = name_yaml_EWC 

    print(config["dataset_params"]["dataset"])
    if not "dataset" in config["dataset_params"] or config["dataset_params"]["dataset"] == "DomainNet": 
        from src.Loaders.DataLoaders import get_loaders
        train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])

    elif config["dataset_params"]["dataset"] == "DN4IL":
        from src.Loaders.DataLoaders_DN4IL import get_loaders
        train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["path_dn4il"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"], config=config)
    else:
        raise ValueError(f'{config["dataset_params"]["dataset"]} Dataset not supported')


    if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities"):
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities")

    idx2domain = val_loader.dataset.idx2domain
    num_domains = val_loader.dataset.num_domains

    model = define_network(config['student']["model_params"]).to(device)
    
    State_Dicts_EWC = [torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_EWC}/model_{idx2domain[i]}_student.pth", map_location=device) for i in range(num_domains)]
    State_Dicts_NO_EWC = [torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_NO_EWC}/model_{idx2domain[i]}_student.pth", map_location=device) for i in range(num_domains)]
    
    distances_EWC = np.zeros((num_domains, num_domains))
    distances_NO_EWC = np.zeros((num_domains, num_domains))

    importance_list_ewc = []
    importance_list_no_ewc = []
    for i in tqdm(range(num_domains)):
        model.load_state_dict(State_Dicts_EWC[i])
        importance_list_ewc.append(compute_importances(model, val_loader, torch.nn.CrossEntropyLoss(), device))
        importances_ewc = add_importances(importance_list_ewc, mean_importances=False)

        model.load_state_dict(State_Dicts_NO_EWC[i])   
        importance_list_no_ewc.append(compute_importances(model, val_loader, torch.nn.CrossEntropyLoss(), device))
        importances_no_ewc = add_importances(importance_list_no_ewc, mean_importances=False)
        
        for j in range(i+1, num_domains):
            distance_EWC = compute_loss(State_Dicts_EWC[i], State_Dicts_EWC[j], importances_ewc)
            distance_NO_EWC = compute_loss(State_Dicts_NO_EWC[i], State_Dicts_NO_EWC[j], importances_no_ewc)
            print(f"EWC: {distance_EWC.item()}")
            print(f"NO EWC: {distance_NO_EWC.item()}")

            # print(f"Euclidean distance between {idx2domain[i]} and {idx2domain[j]} with EWC: {euc_distance_EWC}")
            # print(f"Euclidean distance between {idx2domain[i]} and {idx2domain[j]} without EWC: {euc_distance_NO_EWC}")
            distances_EWC[j, i] = distance_EWC.item()
            distances_EWC[i, j] = distance_EWC.item()

            distances_NO_EWC[j, i] = distance_NO_EWC.item()
            distances_NO_EWC[i, j] = distance_NO_EWC.item()

    labels = [idx2domain[i] for i in range(num_domains)]
    distance_EWC *= 100
    distance_NO_EWC *= 100
    plot_strictly_lower_triangular_heatmap(distances_EWC, labels, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities/similarities_euclidean_EWC.png", "Distances domain models with EWC")

    plot_strictly_lower_triangular_heatmap(distances_NO_EWC, labels, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities/similarities_euclidean_NO_EWC.png", "Distances domain models without EWC")

