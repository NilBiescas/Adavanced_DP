import torch
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
from .plot_utils import training_plot
import numpy as np
from .ewc_functions import compute_importances, add_importances, compute_loss
from .task_vectors import TaskVector


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


def baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch):
    model.train()


    print(f"Epoch {epoch}")
    train_loss = 0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)): 
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    print("Validation")
    val_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloder), total=len(val_dataloder)):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_dataloder)

    return train_loss, val_loss



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


def TeacherStudent_train_epoch(teacher, student, train_dataloader, val_dataloder, optimizer_teacher, optimizer_studen, criterion, device, epoch, domains_trained, alpha_ewc_student, alpha_ewc_teacher, temperature=1):
    # Per entra tecaher_student, el que podem fer es cambiar el criterion en base a si tenen wce o no
    teacher.train()
    student.train()
    assert (teacher.gradient_stop != student.gradient_stop) or ((teacher.gradient_stop == student.gradient_stop) and (student.gradient_stop == False)), "At least one of the models should have gradient_stop=False"
    print(f"Epoch {epoch}")
    teacher_train_total_loss = 0
    student_train_total_loss = 0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        data, target = data[0].to(device), data[1].to(device)
      
        optimizer_teacher.zero_grad()
        optimizer_studen.zero_grad()
        
        teacher_output = teacher(data)
        student_output = student(data)
        
        teacher_loss = compute_criterion(criterion, teacher, teacher_output, target, domains_trained, alpha_ewc_teacher) if not teacher.gradient_stop else 0
        student_loss = compute_criterion(criterion, student, student_output, target, domains_trained, alpha_ewc_student) if not student.gradient_stop else 0

        distillation = distillation_loss(student_output, teacher_output.detach(), T=temperature) if not (teacher.gradient_stop or student.gradient_stop) else criterion(student_output, teacher_output)

        update_moving_average(teacher, student, alpha=0.99) if teacher.gradient_stop else None
        update_moving_average(student, teacher, alpha=0.99) if student.gradient_stop else None
        
        teacher_loss.backward()
        student_loss += distillation
        student_loss.backward()
        
        if not teacher.gradient_stop:
            optimizer_teacher.step()
            
        if not student.gradient_stop:
            optimizer_studen.step()

        teacher_train_total_loss += teacher_loss.item() if not teacher.gradient_stop else 0
        student_train_total_loss += student_loss.item() if not student.gradient_stop else 0
        
    teacher_train_total_loss /= len(train_dataloader)
    student_train_total_loss /= len(train_dataloader)
    
    print("Validation")
    val_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloder), total=len(val_dataloder)):
            data, target = data[0].to(device), data[1].to(device)

            student_output = student(data)
            loss = torch.nn.CrossEntropyLoss()(student_output, target)
            val_loss += loss.item()
            
    val_loss /= len(val_dataloder)
    return teacher_train_total_loss, student_train_total_loss, val_loss



def baseline_train(model, train_dataloader, val_dataloder, test_dataloader, 
                   optimizer, criterion, device, epochs, early_stopping_patience=5, 
                   scheduler_config=None):
    
    print("Training")

    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}", exist_ok=True)
    idx2domain = train_dataloader.dataset.idx2domain
    Domains_trained = 0
    num_domains = train_dataloader.dataset.num_domains
    
    start_lr = optimizer.param_groups[0]['lr']

    prev_accs_top1 = []
    prev_accs_top5 = []
    
    for domain in range(num_domains):
        # Seting the current domain to train and validation dataloaders
        train_dataloader.dataset.select_domain(domain)
        val_dataloder.dataset.select_domain(domain)

        optimizer.param_groups[0]['lr'] = start_lr
        print(start_lr)

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
            train_loss, val_loss = baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch)
            wandb.log({f"train_loss_{idx2domain[domain]}": train_loss, f"val_loss_{idx2domain[domain]}": val_loss, "epoch": epoch})
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
                torch.save(model.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pth")

            elif (early_stopping_patience != -1) and (counter >= early_stopping_patience):
                break
        
        model.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pth"))

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
    
def generate_plot_practica(eval_top1_acc, eval_top5_acc, num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_plot):
    
    avg_task_test_acc = np.mean(eval_top1_acc)
    bar_heights = eval_top1_acc + [0]*(num_domains - len(eval_top1_acc))
    plt.bar(x = range(num_domains), height=bar_heights, zorder=1)
    plt.xticks(range(num_domains), [idx2domain[domain_id] for domain_id in range(num_domains)], rotation='vertical')
    plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
    plt.text(0, avg_task_test_acc+0.002, f'(average)', c=[0.4]*3, size=8)
    
    if prev_accs_top1 is not None:
        # plot the previous step's accuracies on top
        # (will show forgetting in red)
        for p, prev_acc_list in enumerate(prev_accs_top1):
            plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs_top1)))
    
    plt.savefig(save_plot + "_top1.png")
    plt.close()

    avg_task_test_acc = np.mean(eval_top5_acc)
    bar_heights = eval_top5_acc + [0]*(num_domains - len(eval_top5_acc))
    plt.bar(x = range(num_domains), height=bar_heights, zorder=1)
    plt.xticks(range(num_domains), [idx2domain[domain_id] for domain_id in range(num_domains)], rotation='vertical')
    plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
    plt.text(0, avg_task_test_acc+0.002, f'(average)', c=[0.4]*3, size=8)

    if prev_accs_top5 is not None:
        # plot the previous step's accuracies on top
        # (will show forgetting in red)
        for p, prev_acc_list in enumerate(prev_accs_top5):
            plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs_top5)))
    # Log the plot
    plt.savefig(save_plot + "_top5.png")
    plt.close()
    
    top1 = plt.imread(save_plot + "_top1.png")
    top5 = plt.imread(save_plot + "_top5.png")
    wandb.log({f'{os.path.basename(save_plot)}' + "_top1": [wandb.Image(top1, caption="Top1 Accuracy")], 
               f'{os.path.basename(save_plot)}' + "_top5": [wandb.Image(top5, caption="Top5 Accuracy")]})


    
def train_teacher_student(teacher, student, train_dataloader, val_dataloder, 
                          test_dataloader, optimizer_teacher, optimizer_student, criterion, 
                          device, epochs, early_stopping_patience=5, scheduler_config=None, Averaging_importances=False, alpha_ewc_student=1.0, alpha_ewc_teacher=1.0, temperature=1):
    
    print("Alpha EWC Student:", alpha_ewc_student)
    print("Alpha EWC Teacher:", alpha_ewc_teacher)

    print("Temperature", temperature)
    
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{teacher.name}", exist_ok=True)
    idx2domain = train_dataloader.dataset.idx2domain
    Domains_trained = 0
    num_domains = train_dataloader.dataset.num_domains

    start_lr_teacher = optimizer_teacher.param_groups[0]['lr']
    start_lr_student = optimizer_student.param_groups[0]['lr']

    prev_accs_top1 = []
    prev_accs_top5 = []

    list_task_importances_teacher = []
    list_task_importances_student = []
    for domain in range(num_domains):
        # Seting the current domain to train and validation dataloaders
        train_dataloader.dataset.select_domain(domain)
        val_dataloder.dataset.select_domain(domain)

        optimizer_teacher.param_groups[0]['lr'] = start_lr_teacher
        optimizer_student.param_groups[0]['lr'] = start_lr_student

         # Reset the squeduler
        if scheduler_config is not None:
            if scheduler_config["name"] == "StepLR":
                scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
                scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
            elif scheduler_config["name"] == "ReduceLROnPlateau":
                scheduler_teacher = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_teacher, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"], threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
                scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, mode='min', factor=scheduler_config["factor"], patience=scheduler_config["patience"], threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            else:
                raise ValueError(f'{scheduler_config["name"]} Scheduler not supported')

        best_val_loss = float('inf')
        counter = 0
        for epoch in tqdm(range(epochs)):
            teacher_train_total_loss, student_train_total_loss, val_loss = TeacherStudent_train_epoch(teacher, student, train_dataloader, val_dataloder, 
                                                                                                      optimizer_teacher, optimizer_student, 
                                                                                                      criterion, device, epoch, Domains_trained, alpha_ewc_student, alpha_ewc_teacher, temperature=temperature)
            
            wandb.log({f"train_loss_teacher{idx2domain[domain]}": teacher_train_total_loss, 
                       f"student_loss_teacher{idx2domain[domain]}": student_train_total_loss, 
                       f"val_loss_student{idx2domain[domain]}": val_loss, 
                       "epoch": epoch})
        
            train_top1_acc, train_top5_acc = top1_and_top_k_accuracy_domain(student, train_dataloader, device, k=5)
            wandb.log({f"train_top1_acc_student_{idx2domain[domain]}": train_top1_acc, f"train_top5_acc_student_{idx2domain[domain]}": train_top5_acc, "epoch": epoch})
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain(student, val_dataloder, device, k=5)
            wandb.log({f"val_top1_acc_student_{idx2domain[domain]}": val_top1_acc, f"val_top5_acc_student_{idx2domain[domain]}": val_top5_acc, "epoch": epoch})

            if scheduler_config is not None:
                if scheduler_config["name"] == "StepLR":
                    scheduler_teacher.step()
                    scheduler_student.step()
                elif scheduler_config["name"] == "ReduceLROnPlateau":
                    scheduler_teacher.step(val_loss)
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
        if teacher.train_with_ewc:
            teacher_importances = compute_importances(teacher, val_dataloder, criterion, device)
            list_task_importances_teacher.append(teacher_importances)
            teacher_importances = add_importances(list_task_importances_teacher, mean_importances=Averaging_importances) # We sum the importances of all the tasks equally
            teacher._importances = teacher_importances
            teacher._old_model_state_dict = teacher.state_dict()

        if student.train_with_ewc:
            student_importances = compute_importances(student, val_dataloder, criterion, device)
            list_task_importances_student.append(student_importances)
            student_importances = add_importances(list_task_importances_student, mean_importances=Averaging_importances)
            student._importances = student_importances
            student._old_model_state_dict = student.state_dict()
        
        eval_top1_acc = []
        eval_top5_acc = []
        for i in range(domain+1):
            #train_dataloader.dataset.select_domain(i)
            val_dataloder.dataset.select_domain(i)
            test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(student, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_student_{idx2domain[i]}": test_top1_acc, f"top5_acc_prev_student_{idx2domain[i]}": test_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(test_top1_acc); eval_top5_acc.append(test_top5_acc)

        prev_accs_top1.append(eval_top1_acc); prev_accs_top5.append(eval_top5_acc)
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/train_plots", exist_ok=True)
        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/train_plots/num_domains_{Domains_trained}_domain_{idx2domain[i]}"
        generate_plot_practica(eval_top1_acc, eval_top5_acc, val_dataloder.dataset.num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_path)

    result_top1 = []
    result_top5 = []
    for i in range(num_domains):
        test_dataloader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(student, test_dataloader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())
        wandb.log({f"test_top1_acc_student_{idx2domain[domain]}": test_top1_acc})
    
    save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{student.name}/test_{student.name}"
    generate_plot_practica(result_top1, result_top5, test_dataloader.dataset.num_domains, idx2domain, None, None, save_path)

    return teacher, student


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