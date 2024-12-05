import torch
import torch.nn as nn
import torch.optim.optimizer
import torch.utils.data.dataloader
from tqdm import tqdm
import wandb
import os
from .utils.ewc_functions import compute_importances, add_importances, compute_loss


import torch
import torch.nn as nn
import torch.nn.functional as F



from .utils.evaluateFunctions_and_definiOptimizer import top1_and_top_k_accuracy_domain
from .utils.plotting_functions import generate_plot_practica, createCSV, plotStablityPlasticity, plot_strictly_lower_triangular_heatmap

import numpy as np
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def compute_criterion(criterion, model, output, target, domains_trained, alpha=1.0):

    if model.train_with_ewc and domains_trained >= 1:
        return compute_loss(model, model._old_model_state_dict, output, target, model._importances, criterion=criterion, alpha=alpha)
    else:
        return criterion(output, target)


def distillation_loss(student_output, teacher_output, T=1):
    """
    Compute the distillation loss
    """
    return torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(student_output/T, dim=1), torch.nn.functional.softmax(teacher_output/T, dim=1)) * T * T

def update_moving_average(ema_model, student_model, alpha=0.99):
    """ Update the moving average of the teacher's model using the student's weights. """
    with torch.no_grad():
        for ema_param, student_param in zip(ema_model.parameters(), student_model.parameters()):
            ema_param.data = alpha * ema_param.data + (1.0 - alpha) * student_param.data



def TeacherStudent_train_epoch_dino_real(teacher: nn.Module, student: nn.Module, 
                                    train_dataloader: torch.utils.data.dataloader, val_dataloder: torch.utils.data.dataloader, optimizer_studen: torch.optim.Optimizer, 
                                    criterion: nn.CrossEntropyLoss, dino_distillation_loss: nn.Module, device: int, epoch: int, 
                                    domains_trained: int, alpha_ewc_student: float, update_frequency: int =10):
    # Per entra tecaher_student, el que podem fer es cambiar el criterion en base a si tenen wce o no
    teacher.train()
    student.train()
    print(f"Epoch {epoch}")
    student_train_total_loss = 0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        images, target = data
        global_views = torch.cat(images[:2]).to(device)
        local_views = torch.cat(images[2:]).to(device)
        target = target.to(device)
        
        optimizer_studen.zero_grad()
        teacher_output = teacher(global_views)  # only the 2 global views pass through the teacher
        student_output = student(local_views)
        loss = dino_distillation_loss(student_output, teacher_output, epoch)
        
        
        student_output_global_views = student(global_views)
        first_view, second_view = student_output_global_views.chunk(2)
        hard_loss1 = compute_criterion(criterion, student, first_view, target, domains_trained, alpha_ewc_student)
        hard_loss2 = compute_criterion(criterion, student, second_view, target, domains_trained, alpha_ewc_student)
        
        student_loss = loss + hard_loss1 + hard_loss2
        student_loss.backward()
        update_moving_average(teacher, student, alpha=0.99)

        optimizer_studen.step()

        #teacher_train_total_loss += teacher_loss.item() if not teacher.gradient_stop else 0
        student_train_total_loss += student_loss.item()
        
    student_train_total_loss /= len(train_dataloader)
    
    print("Validation")
    val_loss = 0
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloder), total=len(val_dataloder)):
            data, target = data[0].to(device), data[1].to(device)

            student_output = student(data)
            loss = cross_entropy_loss(student_output, target)
            val_loss += loss.item()
            
    val_loss /= len(val_dataloder)
    return student_train_total_loss, val_loss


def top1_and_top_k_accuracy_domain_dino_real(model, loader, device, k=5):
    print("Computing accuracy")
    total = 0
    correct_top1 = 0
    correct_topk = 0
    for i, data in enumerate(loader):
        if len(data) == 2:
            inputs, targets = data
            inputs = inputs[0]

        else:
            inputs, targets = data
        
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
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



def train_teacher_student_DINO_real(teacher: nn.Module, student: nn.Module, 
                               train_dataloader, val_dataloder, test_dataloader, 
                               optimizer_student, criterion, 
                          device,scheduler_config=None, Averaging_importances=False, config=None):
    
    assert config is not None, "Config is required"
        
    alpha_ewc_student= config["student"]["ewc_params"]["lambda"]  # Updated
    alpha_ewc_teacher= config["teacher"]["ewc_params"]["lambda"]  # Updated,
    early_stopping_patience=config["training_params"]["early_stopping_patience"]
    epochs = config["training_params"]["epochs"]
    
    
    upd_frequency = config["student"]['dino_params']["update_frequency"]
    
    print("Alpha EWC Student:", alpha_ewc_student)
    print("Alpha EWC Teacher:", alpha_ewc_teacher)

    
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{teacher.name}", exist_ok=True)
    idx2domain = train_dataloader.dataset.idx2domain
    Domains_trained = 0
    num_domains = train_dataloader.dataset.num_domains

    start_lr_student = optimizer_student.param_groups[0]['lr']

    prev_accs_top1 = []
    prev_accs_top5 = []
    test_prev_accs_top1 = []
    test_prev_accs_top5 = []
      
    list_task_importances_student = []
    
    #dino_loss = DINOLoss(100, teacher_temp=config['teacher']['temperature'],
    #                     student_temp=config['student']['temperature'], center_momentum=0.9).to(device)

    dino_loss = DINOLoss(
        out_dim = 100,
        ncrops = 10,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp = 0.04,
        teacher_temp = 0.04,
        warmup_teacher_temp_epochs = 3,
        nepochs = epochs,
    ).to(device)

    teacher.load_state_dict(student.state_dict())
    teacher.to(device)
    student.to(device)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    for domain in range(num_domains):
        # Seting the current domain to train and validation dataloaders
        train_dataloader.dataset.select_domain(domain)
        val_dataloder.dataset.select_domain(domain)

        optimizer_student.param_groups[0]['lr'] = start_lr_student

        # Reset the squeduler
        if scheduler_config is not None:
            if scheduler_config["name"] == "StepLR":
                scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
            elif scheduler_config["name"] == "ReduceLROnPlateau":
                scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"], threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            else:
                raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')

        best_val_loss = float('inf')
        counter = 0
        for epoch in tqdm(range(epochs)):
            student_train_total_loss, val_loss = TeacherStudent_train_epoch_dino_real(teacher, student, train_dataloader, val_dataloder, 
                                                                                                    optimizer_student, 
                                                                                                    criterion, dino_loss, device, epoch, Domains_trained, alpha_ewc_student, 
                                                                                                    update_frequency=upd_frequency)
            wandb.log({ 
                    f"student_loss_teacher{idx2domain[domain]}": student_train_total_loss, 
                    f"val_loss_student{idx2domain[domain]}": val_loss, 
                    "epoch": epoch})
        
            train_top1_acc, train_top5_acc = top1_and_top_k_accuracy_domain_dino_real(student, train_dataloader, device, k=5)
            wandb.log({f"train_top1_acc_student_{idx2domain[domain]}": train_top1_acc, f"train_top5_acc_student_{idx2domain[domain]}": train_top5_acc, "epoch": epoch})
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain_dino_real(student, val_dataloder, device, k=5)
            wandb.log({f"val_top1_acc_student_{idx2domain[domain]}": val_top1_acc, f"val_top5_acc_student_{idx2domain[domain]}": val_top5_acc, "epoch": epoch})

            if scheduler_config is not None:
                if scheduler_config["name"] == "StepLR":
                    scheduler_student.step()
                elif scheduler_config["name"] == "ReduceLROnPlateau":
                    scheduler_student.step(val_loss)
                else:
                    raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')
            
            counter += 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(student.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/model_{idx2domain[domain]}_student.pth")
                torch.save(teacher.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{teacher.name}/model_{idx2domain[domain]}_teacher.pth")
            elif (early_stopping_patience != -1) and (counter >= early_stopping_patience):
                break
        # Load the best model
        student.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/model_{idx2domain[domain]}_student.pth"))
        teacher.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{teacher.name}/model_{idx2domain[domain]}_teacher.pth"))
        
        Domains_trained += 1   
        # Calculate the importances for this domain
        if student.train_with_ewc:
            student_importances = compute_importances(student, val_dataloder, criterion, device)
            list_task_importances_student.append(student_importances)
            student_importances = add_importances(list_task_importances_student, mean_importances=Averaging_importances)
            student._importances = student_importances
            student._old_model_state_dict = student.state_dict()
        
        eval_top1_acc = []
        eval_top5_acc = []
        # Computing plasticity
        for i in range(domain+1):
            #train_dataloader.dataset.select_domain(i)
            val_dataloder.dataset.select_domain(i)
            test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain_dino_real(student, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_student_{idx2domain[i]}": test_top1_acc, f"top5_acc_prev_student_{idx2domain[i]}": test_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(test_top1_acc); eval_top5_acc.append(test_top5_acc)

        test_top1_acc = []
        test_top5_acc = []
        # Computing stability
        for i in range(domain +1):
            test_dataloader.dataset.select_domain(i)
            test_top1_acc_d, test_top5_acc_d = top1_and_top_k_accuracy_domain_dino_real(student, test_dataloader, device, k=5)
            test_top1_acc.append(test_top1_acc_d.cpu().item()); test_top5_acc.append(test_top5_acc_d.cpu().item())
            
        prev_accs_top1.append(eval_top1_acc); prev_accs_top5.append(eval_top5_acc)
        test_prev_accs_top1.append(test_top1_acc); test_prev_accs_top5.append(test_top5_acc)
        
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/train_plots", exist_ok=True)
        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/train_plots/num_domains_{Domains_trained}_domain_{idx2domain[i]}"
        generate_plot_practica(eval_top1_acc, eval_top5_acc, val_dataloder.dataset.num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_path)


    result_top1 = []
    result_top5 = []
    # Evaluate stability
    for i in range(num_domains):
        test_dataloader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(student, test_dataloader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())
        wandb.log({f"test_top1_acc_student_{idx2domain[domain]}": test_top1_acc})
    
    save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/test_{student.name}"
    generate_plot_practica(result_top1, result_top5, test_dataloader.dataset.num_domains, idx2domain, None, None, save_path)
    
    createCSV(test_prev_accs_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/afterEachdomain_top1_plasticity.csv", idx2domain)
    createCSV(test_prev_accs_top5, f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/afterEachdomain_top5_plasticity.csv", idx2domain)
    
    plotStablityPlasticity(test_prev_accs_top1, result_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/stability_plasticity.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top1, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/heatmap_top1.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top5, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/heatmap_top5.png")
    
    return teacher, student