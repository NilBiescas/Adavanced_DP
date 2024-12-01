import sys
import os
import yaml

from src.utils.task_vectors import TaskVector
from src.utils.train_loops import top1_and_top_k_accuracy_domain

name_yaml = sys.argv[1]

RUN_PATH = f"runs/{name_yaml}"

config = yaml.load(open(f"Setups/{name_yaml}.yaml", "r"), Loader=yaml.FullLoader)



Task_Vectors = [TaskVector(pretrained_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt", 
                                   finetuned_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[i]}.pt") for i in range(num_domains)]

TaskVectors_sum = sum(Task_Vectors)
tmp_model = TaskVectors_sum.apply_to(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt")
model.load_state_dict(tmp_model.state_dict())
del tmp_model
model.eval()

result_top1 = []
result_top5 = []
for i in range(num_domains):
    test_dataloader.dataset.select_domain(i)
    test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_dataloader, device, k=5)
    result_top1.append(test_top1_acc.cpu())
    result_top5.append(test_top5_acc.cpu())
    wandb.log({f"test_top1_acc_{idx2domain[i]}": test_top1_acc, f"test_top5_acc_{idx2domain[i]}": test_top5_acc})

save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/test_{model.name}"
generate_plot_practica(result_top1, result_top5, test_dataloader.dataset.num_domains, idx2domain, None, None, save_path)