import os

# Set GPU to use and order of GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
from tqdm import tqdm
from src.Models.baseline_arch import baseline_with_ewc
from src.Loaders.DataLoaders_DN4IL import DN4IL, partition  
import torch.nn.functional as F
import copy

def compute_loss(model, old_model_state_dict, importances):
    regularitzation_loss = 0
    for name, param in model.named_parameters():
        if name in old_model_state_dict:
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
        log_probs = F.log_softmax(outputs, dim=1)[:, targets].mean()
        log_probs.backward()

        for name, param in model.named_parameters():
            if param.grad.data is not None:
                importances[name] += (param.grad.data.clone() ** 2) / len(data_loader)

        model.zero_grad()

    return importances


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = baseline_with_ewc(num_classes=100)
model.load_state_dict(torch.load("Runs/baseline_dn4il_v2_domain_order/model_clipart.pth", map_location=device))
model.to(device)
domains = ["real", "clipart", "infograph", "painting", "sketch", "quickdraw"]

test_dataloader = None

return2views = False
transform_type = 'default'
dataset = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL",
                partition=partition.TEST, return2views = return2views, domainOrder = domains, transform_type=transform_type)
idx2domain = dataset.idx2domain
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)
print("Computing importances")
importances = compute_importances(model, test_dataloader, torch.nn.CrossEntropyLoss(), device)
print("Computing loss")

model_old = baseline_with_ewc(num_classes=100)
model_old.load_state_dict(torch.load("Runs/baseline_dn4il_v2_domain_order/model_quickdraw.pth", map_location=device))
model_old.to(device)

model.old_model_state_dict = copy.deepcopy(model_old.state_dict())
del model_old

model.zero_grad()
# Print the first importance value
print(list(importances.values())[0])

loss = compute_loss(model, model.old_model_state_dict, importances)
print(loss)
model.zero_grad()
loss.backward()
for name, param in model.named_parameters():
    if param.grad.data is not None:
        print(param.grad.data)
        break