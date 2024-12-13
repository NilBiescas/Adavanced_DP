import torch
from tqdm import tqdm
import wandb
import os
from .utils.evaluateFunctions_and_definiOptimizer import top1_and_top_k_accuracy_domain
from .utils.plotting_functions import plot_strictly_lower_triangular_heatmap, generate_plot_practica

def compute_criterion(criterion, model, output, target, domains_trained=0, alpha=1.0):
    return criterion(output, target)

def compute_per_domain_accuracy(model, dataloader, device, num_domains):
    model.eval()
    domain_correct_top1 = [0] * num_domains
    domain_correct_top5 = [0] * num_domains
    domain_total = [0] * num_domains

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            inputs, targets, domain_labels = data
            inputs, targets, domain_labels = inputs.to(device), targets.to(device), domain_labels.to(device)
            
            outputs = model(inputs)
            
            _, preds_top1 = torch.max(outputs, 1)
            for i in range(num_domains):
                mask = domain_labels == i
                if mask.sum() > 0:
                    domain_correct_top1[i] += (preds_top1[mask] == targets[mask]).sum().item()
                    domain_total[i] += mask.sum().item()

            if model.num_classes >= 5:
                _, preds_top5 = outputs.topk(5, 1, True, True)
                for i in range(num_domains):
                    mask = domain_labels == i
                    if mask.sum() > 0:
                        domain_correct_top5[i] += (preds_top5[mask] == targets[mask].unsqueeze(1)).any(1).sum().item()

    per_domain_acc = []
    per_domain_acc_k5 = []
    for i in range(num_domains):
        if domain_total[i] > 0:
            acc_top1 = domain_correct_top1[i] / domain_total[i] * 100
            acc_top5 = domain_correct_top5[i] / domain_total[i] * 100 if model.num_classes >= 5 else acc_top1
        else:
            acc_top1 = 0.0
            acc_top5 = 0.0

        per_domain_acc.append(acc_top1)
        per_domain_acc_k5.append(acc_top5)

    return per_domain_acc, per_domain_acc_k5


def baseline_train_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device, epoch, alpha=1.0):
    model.train()
    print(f"Epoch {epoch}")
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, targets, domain_labels = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_criterion(criterion, model, outputs, targets, alpha=alpha)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    print("Validation")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            inputs, targets, domain_labels = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    return train_loss, val_loss

def baseline_train_all_domains(
    model, train_dataloader, val_dataloader, test_dataloader,
    optimizer, criterion, device, epochs, early_stopping_patience=5,
    scheduler_config=None, alpha=1.0, **kwargs):

    print("Training on All Domains")
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}", exist_ok=True)

    if isinstance(train_dataloader.dataset, torch.utils.data.Subset):
        num_domains = train_dataloader.dataset.dataset.num_domains
        idx2domain = {i: domain for i, domain in enumerate(train_dataloader.dataset.dataset.domains)}
    else:
        num_domains = train_dataloader.dataset.num_domains
        idx2domain = {i: domain for i, domain in enumerate(train_dataloader.dataset.domains)}

    best_val_loss = float('inf')
    counter = 0

    scheduler = None
    if scheduler_config is not None:
        if scheduler_config["name"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )
        elif scheduler_config["name"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"]
            )
        else:
            raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')

    all_epochs_val_top1 = []
    all_epochs_val_top5 = []

    for epoch in range(epochs):
        train_loss, val_loss = baseline_train_epoch(
            model, train_dataloader, val_dataloader, optimizer, criterion, device, epoch, alpha=alpha
        )

        wandb.log({"train_loss_all_domains": train_loss, "val_loss_all_domains": val_loss, "epoch": epoch})

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/best_model_all_domains.pth")
        else:
            counter += 1
            if (early_stopping_patience != -1) and (counter >= early_stopping_patience):
                print("Early stopping triggered.")
                break

        val_top1_acc, val_top5_acc = compute_per_domain_accuracy(model, val_dataloader, device, num_domains)
        all_epochs_val_top1.append(val_top1_acc)
        all_epochs_val_top5.append(val_top5_acc)

        for domain_idx in range(num_domains):
            wandb.log({
                f"val_top1_acc_{idx2domain[domain_idx]}": val_top1_acc[domain_idx],
                f"val_top5_acc_{idx2domain[domain_idx]}": val_top5_acc[domain_idx],
                "epoch": epoch
            })

        results_dir = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/results"
        os.makedirs(results_dir, exist_ok=True)
        save_plot_path = os.path.join(results_dir, f"val_domain_accuracy_epoch_{epoch}")
        generate_plot_practica(
            eval_top1_acc=val_top1_acc,
            eval_top5_acc=val_top5_acc,
            num_domains=num_domains,
            idx2domain=idx2domain,
            prev_accs_top1=all_epochs_val_top1[:-1] if len(all_epochs_val_top1) > 1 else None,  # Exclude current epoch
            prev_accs_top5=all_epochs_val_top5[:-1] if len(all_epochs_val_top5) > 1 else None,
            save_plot=save_plot_path,
            wandb_exists=True
        )

    model.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/best_model_all_domains.pth"))
    model.eval()

    test_top1_acc, test_top5_acc = compute_per_domain_accuracy(model, test_dataloader, device, num_domains)
    for domain_idx in range(num_domains):
        wandb.log({
            f"test_top1_acc_{idx2domain[domain_idx]}": test_top1_acc[domain_idx],
            f"test_top5_acc_{idx2domain[domain_idx]}": test_top5_acc[domain_idx]
        })

    print(f"Final Test Top-1 Accuracy (per domain): {test_top1_acc}")
    print(f"Final Test Top-5 Accuracy (per domain): {test_top5_acc}")

    plot_strictly_lower_triangular_heatmap(
        data=[test_top1_acc],
        labels=[idx2domain[i] for i in range(num_domains)],
        save_path=os.path.join(results_dir, "heatmap_all_domains_test.png")
    )

    generate_plot_practica(
        eval_top1_acc=test_top1_acc,
        eval_top5_acc=test_top5_acc,
        num_domains=num_domains,
        idx2domain=idx2domain,
        prev_accs_top1=None,
        prev_accs_top5=None,
        save_plot=os.path.join(results_dir, "final_test_domain_accuracy"),
        wandb_exists=True
    )

    return model

