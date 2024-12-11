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

def eval(model, test_loader, device, num_domains, name_plot, name_yaml):
    result_top1 = []
    result_top5 = []
    for i in range(num_domains):
        test_loader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_loader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())

    #save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/Validation/{name_plot}"
    # generate_plot_practica(result_top1, result_top5, test_loader.dataset.num_domains, test_loader.dataset.idx2domain, None, None, save_path, wandb_exists=False)
    
    return np.mean(result_top1)


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


    if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Validation"):
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Validation")

    idx2domain = val_loader.dataset.idx2domain
    num_domains = val_loader.dataset.num_domains

    # Make a sample of 150 images per domain at the validation set
    val_loader.dataset.sample_domains(40)

    test_prev_accs_top1 = []
    test_prev_accs_top5 = []
    for domains2use in range(1, num_domains+1):
        if domains2use != 1:
            Task_Vectors = [TaskVector(pretrained_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/pretrained_model.pt", 
                                            finetuned_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/model_{idx2domain[i]}.pt") for i in range(domains2use)]

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

            plt.plot(list(dict_results.keys()), list(dict_results.values()))
            plt.xlabel("Scaling coef")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.savefig(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Validation/first_{domains2use}_domains_scaling_coef_effect.png")
            plt.close()

        else:
            model = torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/model_{idx2domain[0]}.pt")
            best_scaling_coef = 1

        # Evaluate the best model on the test set for each domain used
        result_top1 = []
        result_top5 = []
        for i in range(domains2use):
            test_loader.dataset.select_domain(i)
            test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_loader, device, k=5)
            result_top1.append(test_top1_acc.cpu().item())
            result_top5.append(test_top5_acc.cpu().item())
        
        test_prev_accs_top1.append(result_top1)
        test_prev_accs_top5.append(result_top5)

        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/Validation/first_{domains2use}_domains_test_scaling_coef_{best_scaling_coef}"
        generate_plot_practica(result_top1, result_top5, test_loader.dataset.num_domains, idx2domain, None, None, save_path, wandb_exists=False)

    createCSV(test_prev_accs_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/afterEachdomain_top1_plasticity.csv", idx2domain)
    createCSV(test_prev_accs_top5, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/afterEachdomain_top5_plasticity.csv", idx2domain)

    plotStablityPlasticity(test_prev_accs_top1, result_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/stability_plasticity.png")

    # import pandas as pd
    # test_prev_accs_top1_pd = pd.read_csv(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/afterEachdomain_top1_plasticity.csv", header=None)
    # test_prev_accs_top5_pd = pd.read_csv(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/afterEachdomain_top5_plasticity.csv", header=None)
    # test_prev_accs_top1_tmp = []
    # test_prev_accs_top5_tmp = []
    # for i in range(1, num_domains+1):
    #     test_prev_accs_top1_tmp.append(list(test_prev_accs_top1_pd.iloc[i, 1:i+1].values))
    #     test_prev_accs_top5_tmp.append(list(test_prev_accs_top5_pd.iloc[i, 1:i+1].values))
    
    # # Convert to floats
    # test_prev_accs_top5 = []
    # test_prev_accs_top1 = []
    # for i in range(num_domains):
    #     test_prev_accs_top5.append([float(x.split("(")[1].split(")")[0]) for x in test_prev_accs_top5_tmp[i]])
    #     test_prev_accs_top1.append([float(x.split("(")[1].split(")")[0]) for x in test_prev_accs_top1_tmp[i]]) 
        

    plot_strictly_lower_triangular_heatmap(test_prev_accs_top1, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/heatmap_top1.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top5, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml_plots}/heatmap_top5.png")