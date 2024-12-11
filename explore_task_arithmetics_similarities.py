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

def cosine_similarity_domain_vetors(v1, v2):
    """Compute the cosine similarity between two task vectors."""
    cosine = 0
    total = 0
    for key in v1.vector:
        if key in v2.vector:
            tmp = torch.dot(v1.vector[key].flatten(), v2.vector[key].flatten())
            norm_v1 = torch.norm(v1.vector[key].flatten())
            norm_v2 = torch.norm(v2.vector[key].flatten())
            cosine += tmp / (norm_v1 * norm_v2)
            total += 1
        else:
            print(f'Warning: key {key} is present in the first task vector but not in the second')
    return cosine / total


def euclidean_distance_domain_vetors(v1, v2):
    """Compute the euclidean distance between two task vectors."""
    euclidean = 0
    total = 0
    for key in v1.vector:
        if key in v2.vector:
            euclidean += torch.norm(v1.vector[key].flatten() - v2.vector[key].flatten())
            total += 1
        else:
            print(f'Warning: key {key} is present in the first task vector but not in the second')
    return euclidean / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_yaml = sys.argv[1]
    config = yaml.load(open(f"Setups/{name_yaml}.yaml", "r"), Loader=yaml.FullLoader)

    name_yaml_plots = name_yaml + "_40"

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

    # Make a sample of 150 images per domain at the validation set
    val_loader.dataset.sample_domains(40)

    Task_Vectors = [TaskVector(pretrained_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/pretrained_model.pt", 
                               finetuned_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/model_{idx2domain[i]}.pt") for i in range(num_domains)]
    
    cos_similarities = np.zeros((num_domains, num_domains))
    euc_similarities = np.zeros((num_domains, num_domains))

    for i in tqdm(range(num_domains)):
        for j in range(i+1, num_domains):
            cos_similarities[i, j] = cosine_similarity_domain_vetors(Task_Vectors[i], Task_Vectors[j])
            cos_similarities[j, i] = cos_similarities[i, j]

            euc_similarities[i, j] = euclidean_distance_domain_vetors(Task_Vectors[i], Task_Vectors[j])
            euc_similarities[j, i] = euc_similarities[i, j]

    cos_similarities *= 100
    labels = [idx2domain[i] for i in range(num_domains)]
    plot_strictly_lower_triangular_heatmap(cos_similarities, labels, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities/similarities_cosine.png", "Cosine Similarity (%)")

    plot_strictly_lower_triangular_heatmap(euc_similarities, labels, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Similarities/similarities_euclidean.png", "Euclidean Distance")

