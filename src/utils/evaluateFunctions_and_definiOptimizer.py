import torch

def define_optimizer(model, optimizer_config):
    if optimizer_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"])
    elif optimizer_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["lr"])
    elif optimizer_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config["lr"])
    else:
        raise ValueError(f'{optimizer_config["optimizer"]} Optimizer not supported')
    return optimizer

def top1_and_top_k_accuracy_domain(model, loader, device, k=5):
    print("Computing accuracy")
    total = 0
    correct_top1 = 0
    correct_topk = 0
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)

        predicts_top1 = torch.max(outputs, dim=1)[1]  # [bs]
        predicts_topk = torch.topk(outputs, k=k, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

        correct_top1 += (predicts_top1 == targets).sum()

        
        for j in range(k):
            correct_topk += (predicts_topk[:, j] == targets).sum()
        
        total += len(targets)

    top1_acc = (correct_top1 / total).cpu()
    topk_acc = (correct_topk / total).cpu()
    return top1_acc*100, topk_acc*100

