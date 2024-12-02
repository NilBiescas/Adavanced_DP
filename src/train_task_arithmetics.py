import torch
from tqdm import tqdm
import wandb
import os

from .utils.task_vectors import TaskVector


# Import plotting functions

from .utils.plotting_functions import plot_strictly_lower_triangular_heatmap, createCSV, plotStablityPlasticity, generate_plot_practica
from .utils.evaluateFunctions_and_definiOptimizer import define_optimizer, top1_and_top_k_accuracy_domain
from .train_loops import baseline_train_epoch

def Task_Arithmetics_Train(model, train_dataloader, val_dataloder, test_dataloader, 
                   optimizer_config, criterion, device, epochs, early_stopping_patience=5, 
                   scheduler_config=None):
    
    print("Training")

    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}", exist_ok=True)

    torch.save(model, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt")

    idx2domain = train_dataloader.dataset.idx2domain
    Domains_trained = 0
    num_domains = train_dataloader.dataset.num_domains


    prev_accs_top1 = []
    prev_accs_top5 = []
    
    for domain in range(num_domains):
        # Seting the current domain to train and validation dataloaders
        train_dataloader.dataset.select_domain(domain)
        val_dataloder.dataset.select_domain(domain)

        # Reset the model and optimizer
        model.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt").state_dict())
        optimizer = define_optimizer(model, optimizer_config)
        model.train()

        # Reset the squeduler
        if scheduler_config is not None:
            if scheduler_config["name"] == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
            elif scheduler_config["name"] == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"], threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            else:
                raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')

        best_val_loss = float('inf')
        counter = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss, val_loss = baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch)
            wandb.log({f"train_loss_{idx2domain[domain]}": train_loss, f"val_loss_{idx2domain[domain]}": val_loss, "epoch": epoch})
            model.eval()
            train_top1_acc, train_top5_acc = top1_and_top_k_accuracy_domain(model, train_dataloader, device, k=5)
            wandb.log({f"train_top1_acc_{idx2domain[domain]}": train_top1_acc, f"train_top5_acc_{idx2domain[domain]}": train_top5_acc, "epoch": epoch})
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain(model, val_dataloder, device, k=5)
            wandb.log({f"val_top1_acc_{idx2domain[domain]}": val_top1_acc, f"val_top5_acc_{idx2domain[domain]}": val_top5_acc, "epoch": epoch})

            if scheduler_config is not None:
                if scheduler_config["name"] == "StepLR":
                    scheduler.step()
                elif scheduler_config["name"] == "ReduceLROnPlateau":
                    scheduler.step(val_loss)
                else:
                    raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')
                
            counter += 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pt")

            elif (early_stopping_patience != -1) and (counter >= early_stopping_patience):
                break
        
        Task_Vectors = [TaskVector(pretrained_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt", 
                                   finetuned_checkpoint=f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[i]}.pt") for i in range(domain+1)]
        TaskVectors_sum = sum(Task_Vectors)
        tmp_model = TaskVectors_sum.apply_to(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/pretrained_model.pt")
        model.load_state_dict(tmp_model.state_dict())
        del tmp_model
        model.eval()

        Domains_trained += 1
        eval_top1_acc = []
        eval_top5_acc = []
        for i in range(domain+1):
            val_dataloder.dataset.select_domain(i)
            test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_{idx2domain[i]}": test_top1_acc, f"top5_acc_prev_{idx2domain[i]}": test_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(test_top1_acc)
            eval_top5_acc.append(test_top5_acc)
        
        prev_accs_top1.append(eval_top1_acc); prev_accs_top5.append(eval_top5_acc)
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots", exist_ok=True)
        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots/num_domains_{Domains_trained}_domain_{idx2domain[i]}"
        generate_plot_practica(eval_top1_acc, eval_top5_acc, val_dataloder.dataset.num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_path)
    

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
    return model
