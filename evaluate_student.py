import os

# Set GPU to use and order of GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import torch
from src.utils.plotting_functions import generate_plot_practica, createCSV, plotStablityPlasticity, plot_strictly_lower_triangular_heatmap
from src.utils.evaluateFunctions_and_definiOptimizer import top1_and_top_k_accuracy_domain
from src.Loaders.DataLoaders_DN4IL import DN4IL, partition
import sys
from tqdm import tqdm
import yaml

yaml_name = sys.argv[1]
print(f"YAML Name: {yaml_name}")
with open(f"/fhome/amlai07/Adavanced_DP/Setups/{yaml_name}.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
runs_dir = "/fhome/amlai07/Adavanced_DP/Runs"



if "domain_order" in config['dataset_params']:
  domains = config['dataset_params']['domain_order']
else:
  domains = None


test_prev_accs_top1 = []
test_prev_accs_top5 = []
result_top1 = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataloader = None


return2views = False
transform_type = 'default'
dataset = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL",
                partition=partition.TEST, return2views = return2views, domainOrder = domains, transform_type=transform_type)
idx2domain = dataset.idx2domain
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

from src.Models.baseline_arch import baseline_with_ewc


student = baseline_with_ewc(num_classes=100, train_with_ewc=config['student']['model_params']['train_with_ewc'])
for idx, domain in enumerate(dataset.domains):

    if "Student" in yaml_name:
        student_weights_path = f"{runs_dir}/{yaml_name}/model_{domain}_student.pth"
    else:
        student_weights_path = f"{runs_dir}/{yaml_name}/model_{domain}.pth"
        
    teacher_weights = torch.load(student_weights_path)
    student.load_state_dict(teacher_weights)
    student.to(device)
    student.eval()
    # Evaluate stability
    test_top1_acc = []
    test_top5_acc = []
    for i in range(idx + 1):
        print(f"Domain: {i} with weights: {student_weights_path}")
        test_dataloader.dataset.select_domain(i)
        test_top1_acc_d, test_top5_acc_d = top1_and_top_k_accuracy_domain(student, test_dataloader, device, k=5)
        test_top1_acc.append(test_top1_acc_d.cpu().item()); test_top5_acc.append(test_top5_acc_d.cpu().item())
    
    print(test_prev_accs_top1, test_top5_acc)
    test_prev_accs_top1.append(test_top1_acc); test_prev_accs_top5.append(test_top5_acc)

result_top1 = []
# Evaluate plasticity
for i in range(len(dataset.domains)):
    test_dataloader.dataset.select_domain(i)
    test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(student, test_dataloader, device, k=5)
    result_top1.append(test_top1_acc.cpu())
import os
os.makedirs(f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn", exist_ok=True)
createCSV(test_prev_accs_top1, f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn/afterEachdomain_top1_plasticity.csv", idx2domain)
createCSV(test_prev_accs_top5, f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn/afterEachdomain_top5_plasticity.csv", idx2domain)

plotStablityPlasticity(test_prev_accs_top1, result_top1, f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn/stability_plasticity.png")
plot_strictly_lower_triangular_heatmap(test_prev_accs_top1, list(idx2domain.values()), f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn/heatmap_top1.png")
plot_strictly_lower_triangular_heatmap(test_prev_accs_top5, list(idx2domain.values()), f"{runs_dir}/{yaml_name}/EvaluationStudent_orderItWasTrainedOn/heatmap_top5.png")
