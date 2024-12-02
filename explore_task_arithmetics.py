import sys
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm

from src.utils.task_vectors import TaskVector
from src.utils.evaluateFunctions_and_definiOptimizer import top1_and_top_k_accuracy_domain
from src.utils.plotting_functions import generate_plot_practica

def eval(model, test_loader, device, num_domains, name_plot, name_yaml):
    result_top1 = []
    result_top5 = []
    for i in range(num_domains):
        test_loader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_loader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())

    if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/Validation"):
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/Validation")

    save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/Validation/{name_plot}"
    generate_plot_practica(result_top1, result_top5, test_loader.dataset.num_domains, test_loader.dataset.idx2domain, None, None, save_path, wandb_exists=False)
    
    return np.mean(result_top1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_yaml = sys.argv[1]
    config = yaml.load(open(f"Setups/{name_yaml}.yaml", "r"), Loader=yaml.FullLoader)

    print(config["dataset_params"]["dataset"])
    if not "dataset" in config["dataset_params"] or config["dataset_params"]["dataset"] == "DomainNet": 
        from src.Loaders.DataLoaders import get_loaders
        train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])

    elif config["dataset_params"]["dataset"] == "DN4IL":
        from src.Loaders.DataLoaders_DN4IL import get_loaders
        train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["path_dn4il"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])
    else:
        raise ValueError(f'{config["dataset_params"]["dataset"]} Dataset not supported')

    idx2domain = val_loader.dataset.idx2domain
    num_domains = val_loader.dataset.num_domains

    Task_Vectors = [TaskVector(pretrained_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/pretrained_model.pt", 
                                    finetuned_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/model_{idx2domain[i]}.pt") for i in range(num_domains)]

    TaskVectors_sum = sum(Task_Vectors)
    dict_results = {}
    for i in tqdm(range(1, 101, 1)):
        name_plot = f"scaling_coef_{i/100}"
        model = TaskVectors_sum.apply_to(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/pretrained_model.pt", scaling_coef=i/100)

        dict_results[i] = eval(model, val_loader, device, num_domains, name_plot, name_yaml)
        print(f"Scaling coef: {i/100}, Accuracy: {dict_results[i]}")
    
    best_scaling_coef = max(dict_results, key=dict_results.get)/100
    print(f"Best scaling coef: {best_scaling_coef}")

    model = TaskVectors_sum.apply_to(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/pretrained_model.pt", scaling_coef=best_scaling_coef)

    # Evaluate the best model on the test set
    result_top1 = []
    result_top5 = []
    for i in range(num_domains):
        test_loader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_loader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())

    save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/test_{name_yaml}_scaling_coef_{best_scaling_coef}"
    generate_plot_practica(result_top1, result_top5, test_loader.dataset.num_domains, idx2domain, None, None, save_path, wandb_exists=False)