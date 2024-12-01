import torch
import torch.nn as nn

def compute_loss(model, old_model_state_dict, prediction, target, importances, criterion=torch.nn.CrossEntropyLoss(), alpha=1.0):
    loss = criterion(prediction, target)    
    
    regularitzation_loss = 0
    for name, param in model.named_parameters():
        if name in old_model_state_dict:
            regularitzation_loss += (importances[name] * (param - old_model_state_dict[name]).pow(2)).sum() 
    
    return loss + alpha * regularitzation_loss

def add_importances(list_task_importances, mean_importances=False):
    importances = {}
    for task_importances in list_task_importances:
        for name, importance in task_importances.items():
            if name in importances:
                importances[name] += importance
            else:
                importances[name] = importance

    if mean_importances:
        importances = {name: importance / len(list_task_importances) for name, importance in importances.items()}
    return importances


def compute_importances(model, data_loader, criterion, device):
    model.eval()
    model.zero_grad()
    importances = {}
    for name, param in model.named_parameters():
        importances[name] = torch.zeros_like(param)

    for inputs, targets in data_loader:
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