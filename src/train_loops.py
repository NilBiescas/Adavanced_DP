import torch
from tqdm import tqdm
import wandb
import os

from .utils.ewc_functions import compute_importances, add_importances, compute_loss, compute_importances_v2, compute_importances_v3, recompute_importances
from .utils.task_vectors import TaskVector
import copy


# Import plotting functions

from .utils.plotting_functions import plot_strictly_lower_triangular_heatmap, createCSV, plotStablityPlasticity, generate_plot_practica
from .utils.evaluateFunctions_and_definiOptimizer import define_optimizer, top1_and_top_k_accuracy_domain


def compute_criterion(criterion, model, output, target, domains_trained, alpha=1.0):

    if model.train_with_ewc and domains_trained >= 1:

        return compute_loss(model, model._old_model_state_dict, output, target, model._importances, criterion=criterion, alpha=alpha)
    else:
        return criterion(output, target)

def computeCriterionTeacherStudent(criterion, student, teacher, output_student, output_teacher, target, domains_trained, alpha_ewc_student, alpha_ewc_teacher, temperature=1):

    distill_loss = distillation_loss(output_student, output_teacher.detach(), T=temperature)

    if student.train_with_ewc and domains_trained >= 1:
        loss_student = compute_loss(student, student._old_model_state_dict, output_student, target, student._importances, criterion=criterion, alpha=alpha_ewc_student, distillation_loss=distill_loss)
    else:
        loss_student = criterion(output_student, target) + distill_loss

    if teacher.train_with_ewc and domains_trained >= 1:
        loss_teacher = compute_loss(teacher, teacher._old_model_state_dict, output_teacher, target, teacher._importances, criterion=criterion, alpha=alpha_ewc_teacher, distillation_loss=None)
    else:
        loss_teacher = criterion(output_teacher, target)

    return loss_student, loss_teacher

    

def baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch, domains_trained, alpha):
    model.train()


    print(f"Epoch {epoch}")
    train_loss = 0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)): 
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = compute_criterion(criterion, model, output, target, domains_trained, alpha=alpha)
        
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


def distillation_loss(student_output, teacher_output, T=1):
    """
    Compute the distillation loss
    """
    return torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(student_output/T, dim=1), torch.nn.functional.softmax(teacher_output/T, dim=1)) * T * T


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
        
        teacher_loss, student_loss = computeCriterionTeacherStudent(criterion, student, teacher, student_output, teacher_output, target, domains_trained, alpha_ewc_student, alpha_ewc_teacher, temperature=temperature)
        
        teacher_loss.backward()
        student_loss.backward()
        
        if not teacher.gradient_stop:
            optimizer_teacher.step()
            
        if not student.gradient_stop:
            optimizer_studen.step()

        teacher_train_total_loss += teacher_loss.item()
        student_train_total_loss += student_loss.item()
        
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
                   scheduler_config=None, alpha=1.0, **kwargs):
    
    print("Training")

    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}", exist_ok=True)
    idx2domain = train_dataloader.dataset.idx2domain
    Domains_trained = 0
    num_domains = train_dataloader.dataset.num_domains
    
    start_lr = optimizer.param_groups[0]['lr']

    prev_accs_top1 = []
    prev_accs_top5 = []
    test_prev_accs_top1 = []
    test_prev_accs_top5 = []
    
    if model.train_with_ewc: # Per si volem probar baseline amb ewc
        if ("mean_importances" not in kwargs["training_params"]) or (kwargs["training_params"]["mean_importances"] == True):
            print("Using mean importances")
            print("WARNING: This behaviour does not work as expected")
            Averaging_importances = True
        else:
            print("Not using mean importances")
            Averaging_importances = False
            
        list_task_importances_student = [] 
        
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
            train_loss, val_loss = baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch, Domains_trained, alpha)
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
        
        Domains_trained += 1
        
        if model.train_with_ewc: # Save only if we are using ewc
            importances_v2 = kwargs.get("importances_v2", False)
            importances_v3 = kwargs.get("importances_v3", False)
            if importances_v2:
                print("Using importances_v2")
                student_importances = compute_importances_v2(model, val_dataloder, criterion, device)
            elif importances_v3:
                print("Using importances_v3")
                student_importances = compute_importances_v3(model, val_dataloder, device)
            else:
                student_importances = compute_importances(model, val_dataloder, criterion, device)

            list_task_importances_student.append(student_importances)
            student_importances = add_importances(list_task_importances_student, mean_importances=Averaging_importances)
            model._importances = student_importances
            model._old_model_state_dict = model.state_dict()
        
        model.load_state_dict(torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pth"))
        model.eval()

        ### Evaluate plasticty ###
        eval_top1_acc = []
        eval_top5_acc = []
        for i in range(domain+1):
            val_dataloder.dataset.select_domain(i)
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain(model, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_{idx2domain[i]}": val_top1_acc, f"top5_acc_prev_{idx2domain[i]}": val_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(val_top1_acc)
            eval_top5_acc.append(val_top5_acc)
        prev_accs_top1.append(eval_top1_acc); prev_accs_top5.append(eval_top5_acc) # Save the results

        ### Evaluate plasticty ###
        test_top1_acc = []
        test_top5_acc = []
        for i in range(domain +1):
            test_dataloader.dataset.select_domain(i)
            test_top1_acc_d, test_top5_acc_d = top1_and_top_k_accuracy_domain(model, test_dataloader, device, k=5)
            test_top1_acc.append(test_top1_acc_d.cpu().item()); test_top5_acc.append(test_top5_acc_d.cpu().item())
        test_prev_accs_top1.append(test_top1_acc); test_prev_accs_top5.append(test_top5_acc) # Save the results

        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots", exist_ok=True)
        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots/num_domains_{Domains_trained}_domain_{idx2domain[i]}"
        generate_plot_practica(eval_top1_acc, eval_top5_acc, val_dataloder.dataset.num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_path)
    
    result_top1 = []
    result_top5 = []
    # Evaluate plasticity
    for i in range(num_domains):
        test_dataloader.dataset.select_domain(i)
        test_top1_acc, test_top5_acc = top1_and_top_k_accuracy_domain(model, test_dataloader, device, k=5)
        result_top1.append(test_top1_acc.cpu())
        result_top5.append(test_top5_acc.cpu())
        wandb.log({f"test_top1_acc_{idx2domain[i]}": test_top1_acc, f"test_top5_acc_{idx2domain[i]}": test_top5_acc})

    save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/test_{model.name}"
    generate_plot_practica(result_top1, result_top5, test_dataloader.dataset.num_domains, idx2domain, None, None, save_path)
    
    createCSV(test_prev_accs_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/afterEachdomain_top1_plasticity.csv", idx2domain)
    createCSV(test_prev_accs_top5, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/afterEachdomain_top5_plasticity.csv", idx2domain)
    
    plotStablityPlasticity(test_prev_accs_top1, result_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/stability_plasticity.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top1, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/heatmap_top1.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top5, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/heatmap_top5.png")
    
    return model
    

    
def train_teacher_student(teacher, student, train_dataloader, val_dataloder, 
                          test_dataloader, optimizer_teacher, optimizer_student, criterion, 
                          device, epochs, early_stopping_patience=5, scheduler_config=None, Averaging_importances=False, config=None):
    
    
    alpha_ewc_student= config["student"]["ewc_params"]["lambda"]  # Updated
    alpha_ewc_teacher= config["teacher"]["ewc_params"]["lambda"]  # Updated,
    temperature      = config["training_params"]["temperature"]
    early_stopping_patience=config["training_params"]["early_stopping_patience"]
    epochs = config["training_params"]["epochs"]
    
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
    test_prev_accs_top1 = []
    test_prev_accs_top5 = []
    
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
            
        if (config is not None) and ("warm_start" in config["teacher"]) and (config["teacher"]["warm_start"]["warm_start"]):
            print(f"Doing a warm start for the teacher, in domain {idx2domain[domain]} for {config['teacher']['warm_start']['epochs']} epochs")
            teacher.train()
            for epoch in range(config["teacher"]["warm_start"]["epochs"]):
                train_loss, val_loss = baseline_train_epoch(teacher, train_dataloader, val_dataloder, optimizer_teacher, criterion, device, epoch, Domains_trained, alpha_ewc_teacher)
                wandb.log({f"train_loss_teacher{idx2domain[domain]}": train_loss, f"val_loss_teacher{idx2domain[domain]}": val_loss, "warm_epoch": epoch})

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
        student.eval()
        teacher.eval()
         
        Domains_trained += 1   
        # Calculate the importances for this domain
        if teacher.train_with_ewc:
            importances_v2 = config.get("importances_v2", False)
            importances_v3 = config.get("importances_v3", False)
            recomputed_importances = config.get("recompute_importances", False)
            if recomputed_importances:
                print("Recomputing importances")
                list_task_importances_teacher = recompute_importances(teacher, val_dataloder, device, domain)
            else:
                if importances_v2:
                    print("Using importances_v2")
                    teacher_importances = compute_importances_v2(teacher, val_dataloder, criterion, device)
                elif importances_v3:
                    print("Using importances_v3")
                    teacher_importances = compute_importances_v3(teacher, val_dataloder, device)
                else:
                    teacher_importances = compute_importances(teacher, val_dataloder, criterion, device)
                list_task_importances_teacher.append(teacher_importances)
            
            teacher_importances = add_importances(list_task_importances_teacher, mean_importances=Averaging_importances) # We sum the importances of all the tasks equally
            teacher._importances = teacher_importances
            teacher._old_model_state_dict = copy.deepcopy(teacher.state_dict())

        if student.train_with_ewc:
            importances_v2 = config.get("importances_v2", False)
            importances_v3 = config.get("importances_v3", False)
            recomputed_importances = config.get("recompute_importances", False)
            if recomputed_importances:
                print("Recomputing importances")
                list_task_importances_student = recompute_importances(student, val_dataloder, device, domain)
            else:
                if importances_v2:
                    print("Using importances_v2")
                    student_importances = compute_importances_v2(student, val_dataloder, criterion, device)
                elif importances_v3:
                    print("Using importances_v3")
                    student_importances = compute_importances_v3(student, val_dataloder, device)
                else:
                    student_importances = compute_importances(student, val_dataloder, criterion, device)
                list_task_importances_student.append(student_importances)
                
                list_task_importances_student.append(student_importances)
            
            print(len(list_task_importances_student))
            student_importances = add_importances(list_task_importances_student, mean_importances=Averaging_importances)
            student._importances = student_importances
            student._old_model_state_dict = copy.deepcopy(student.state_dict())
        
        eval_top1_acc = []
        eval_top5_acc = []

        for i in range(domain+1):
            val_dataloder.dataset.select_domain(i)
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain(student, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_student_{idx2domain[i]}": val_top1_acc, f"top5_acc_prev_student_{idx2domain[i]}": val_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(val_top1_acc); eval_top5_acc.append(val_top5_acc)

        test_top1_acc = []
        test_top5_acc = []
        # Evaluate plasticty
        for i in range(domain +1):
            test_dataloader.dataset.select_domain(i)
            test_top1_acc_d, test_top5_acc_d = top1_and_top_k_accuracy_domain(student, test_dataloader, device, k=5)
            test_top1_acc.append(test_top1_acc_d.cpu().item()); test_top5_acc.append(test_top5_acc_d.cpu().item())
        # Append to the eval list that tracks the results over the domains
        prev_accs_top1.append(eval_top1_acc)
        prev_accs_top5.append(eval_top5_acc)
        # Append to the test list that tracks the results over the domains
        test_prev_accs_top1.append(test_top1_acc)
        test_prev_accs_top5.append(test_top5_acc)
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


def BaselineV2(model, train_dataloader, val_dataloder, test_dataloader, 
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
    test_prev_accs_top1 = []
    test_prev_accs_top5 = []
    
    for domain in range(num_domains):
        # Seting the current domain to train and validation dataloaders
        train_dataloader.dataset.select_domain(domain)
        val_dataloder.dataset.select_domain(domain)

        # Reset the optimizer
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
            train_loss, val_loss = baseline_train_epoch(model, train_dataloader, val_dataloder, optimizer, criterion, device, epoch, 0, 0) # We are not using EWC so last two parameters are not used
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
        
        
        model = torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pt")
        model.eval()

        Domains_trained += 1
        ### Evaluate plasticty ###
        eval_top1_acc = []
        eval_top5_acc = []
        for i in range(domain+1):
            val_dataloder.dataset.select_domain(i)
            val_top1_acc, val_top5_acc = top1_and_top_k_accuracy_domain(model, val_dataloder, device, k=5)
            wandb.log({f"top1_acc_prev_{idx2domain[i]}": val_top1_acc, f"top5_acc_prev_{idx2domain[i]}": val_top5_acc, "Trained_domains":Domains_trained})
            eval_top1_acc.append(val_top1_acc)
            eval_top5_acc.append(val_top5_acc)
        prev_accs_top1.append(eval_top1_acc); prev_accs_top5.append(eval_top5_acc) # Save the results

        ### Evaluate plasticty ###
        test_top1_acc = []
        test_top5_acc = []
        for i in range(domain +1):
            test_dataloader.dataset.select_domain(i)
            test_top1_acc_d, test_top5_acc_d = top1_and_top_k_accuracy_domain(model, test_dataloader, device, k=5)
            test_top1_acc.append(test_top1_acc_d.cpu().item()); test_top5_acc.append(test_top5_acc_d.cpu().item())
        test_prev_accs_top1.append(test_top1_acc); test_prev_accs_top5.append(test_top5_acc) # Save the results

        print("Eval test top1 acc")
        print(test_prev_accs_top1)
        print("Eval test top5 acc")
        print(test_prev_accs_top5)
        os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots", exist_ok=True)
        save_path = f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/train_plots/num_domains_{Domains_trained}_domain_{idx2domain[i]}"
        generate_plot_practica(eval_top1_acc, eval_top5_acc, val_dataloder.dataset.num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_path)
    

    model = torch.load(f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/model_{idx2domain[domain]}.pt")
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
    
    createCSV(test_prev_accs_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/afterEachdomain_top1_plasticity.csv", idx2domain)
    createCSV(test_prev_accs_top5, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/afterEachdomain_top5_plasticity.csv", idx2domain)
    
    plotStablityPlasticity(test_prev_accs_top1, result_top1, f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/stability_plasticity.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top1, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/heatmap_top1.png")
    plot_strictly_lower_triangular_heatmap(test_prev_accs_top5, list(idx2domain.values()), f"/fhome/amlai07/Adavanced_DP/Runs/{model.name}/heatmap_top5.png")
    return model
