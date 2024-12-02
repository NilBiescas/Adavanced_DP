import sys
import os
import yaml
import wandb
import torch

from src.Models import define_network
from src.Loaders.DataLoaders import get_loaders

# Train loops

from src.train_loops import train_teacher_student
from src.train_dino import train_teacher_student_DINO
from src.utils.evaluateFunctions_and_definiOptimizer import define_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name_yaml = sys.argv[1]

with open(f"/fhome/amlai07/Adavanced_DP/Setups/{name_yaml}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

teacher = define_network(config['teacher']["model_params"]).to(device)
student = define_network(config['student']["model_params"]).to(device)

print(config["dataset_params"]["dataset"])
if not "dataset" in config["dataset_params"] or config["dataset_params"]["dataset"] == "DomainNet": 
    from src.Loaders.DataLoaders import get_loaders
    train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])

elif config["dataset_params"]["dataset"] == "DN4IL":
    from src.Loaders.DataLoaders_DN4IL import get_loaders
    train_loader, val_loader, test_loader = get_loaders(config["dataset_params"]["data_path"], config["dataset_params"]["path_dn4il"], config["dataset_params"]["image_size"], config["dataset_params"]["batch_size"])
else:
    raise ValueError(f'{config["dataset_params"]["dataset"]} Dataset not supported')

wandb.init(project="Advanced_DP", name=name_yaml)
wandb.config.update(config)


if config["training_params"]["criterion"] == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
else:
    raise ValueError(f'{config["training_params"]["criterion"]} Criterion not supported')

if "scheduler" not in config["training_params"] or config["training_params"]["scheduler"] == "None":
    config_scheduler = None
else:
    config_scheduler = config["training_params"]["scheduler"]

if "early_stopping_patience" not in config["training_params"]:
    config["training_params"]["early_stopping_patience"] = -1


optimizer_teacher = define_optimizer(teacher, config['teacher'])
optimizer_student = define_optimizer(student, config['student'])

teacher.name = name_yaml
student.name = name_yaml

if ("mean_importances" not in config["training_params"]) or (config["training_params"]["mean_importances"] == True):
    print("Using mean importances")
    print("WARNING: This behaviour does not work as expected")
    mean_importances = True
else:
    print("Not using mean importances")
    mean_importances = False


if ("Approach" not in config["training_params"]) or ("TeacherStudent" == config["training_params"]["Approach"]):
    print("Training Teacher Student")
    teacher, student = train_teacher_student(teacher,student, 
                                train_loader, val_loader, test_loader, 
                                optimizer_teacher, optimizer_student,
                                criterion, 
                                device,
                                scheduler_config=config_scheduler,
                                Averaging_importances=mean_importances,
                                config=config)
    

elif "DinoTeacherStudent" == config["training_params"]["Approach"]:
    print("Training DINO Teacher Student")
    teacher, student = train_teacher_student_DINO(teacher,student, 
                                train_loader, val_loader, test_loader, 
                                optimizer_student,
                                criterion, 
                                device,
                                scheduler_config=config_scheduler,
                                Averaging_importances=mean_importances,
                                config=config)
else:
    raise ValueError(f'{config["training_params"]["Approach"]} Approach not supported')

wandb.finish()

# Save the model
if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs"):
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs")

if not os.path.exists(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}"):
    os.makedirs(f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}")

torch.save(teacher.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/teacher.pth")
torch.save(student.state_dict(), f"/fhome/amlai07/Adavanced_DP/Runs/{name_yaml}/student.pth")
