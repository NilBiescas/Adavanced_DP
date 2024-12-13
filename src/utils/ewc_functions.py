import torch
import torch.nn as nn
import torch.nn.functional as F 

def compute_loss(model, old_model_state_dict, prediction, target, importances, criterion=torch.nn.CrossEntropyLoss(), alpha=1.0, distillation_loss=None):
    loss = criterion(prediction, target)    
    
    regularitzation_loss = 0
    for name, param in model.named_parameters():
        if name in old_model_state_dict:
            regularitzation_loss += (importances[name] * (param - old_model_state_dict[name]).pow(2)).sum() 

    if distillation_loss is not None:
        regular_loss = loss + distillation_loss
    else:
        regular_loss = loss
    
    return regular_loss + alpha * regularitzation_loss

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
    """
    Compute the importance of each parameter in the model, 
    Using an aproximation of the fisher information matrix with cross entropy loss.
    """
    model.eval()
    model.zero_grad()
    importances = {}
    for name, param in model.named_parameters():
        print(name)
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


def compute_importances_v2(model, data_loader, criterion, device):
    """
    Compute the importance of each parameter in the model, 
    Using an aproximation of the fisher information matrix with cross entropy loss.
    """
    model.eval()
    model.zero_grad()
    importances = {}
    for name, param in model.named_parameters():
        importances[name] = torch.zeros_like(param)

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        label = torch.argmax(outputs, dim=1)
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), label)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad.data is not None:
                importances[name] += (param.grad.data ** 2) / len(data_loader)

        model.zero_grad()

    return importances

def compute_importances_v3(model, data_loader, device):
    """
    Compute the importance of each parameter in the model, Using the fisher information matrix.
    """
    model.eval()
    model.zero_grad()
    importances = {}
    for name, param in model.named_parameters():
        importances[name] = torch.zeros_like(param)

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=1)[:, targets].mean()
        log_probs.backward()

        for name, param in model.named_parameters():
            if param.grad.data is not None:
                importances[name] += (param.grad.data.clone() ** 2) / len(data_loader)

        model.zero_grad()

    return importances


def recompute_importances(model, data_loader, device, domains):
    """
    Recompute importances for all the previous domain given the current model.
    It uses the fisher information matrix.
    """
    model.eval()
    list_importances = []
    for i in range(domains+1):
        data_loader.dataset.select_domain(i)
        importances = compute_importances_v3(model, data_loader, device)
        list_importances.append(importances)
    
    return list_importances
