import csv
import os
import yaml
import pandas as pd


runs_dir = "/fhome/amlai07/Adavanced_DP/Runs"

yaml_dir = "/fhome/amlai07/Adavanced_DP/Setups"



results = []

for dirpath_ in os.listdir(runs_dir):
    try:
        dirpath = os.path.join(runs_dir, dirpath_)
        if any(dirpath.endswith(ext) for ext in [".txt", "Runs", "train_plots"]):
            print(dirpath)
            continue
        if "afterEachdomain_top1_plasticity.csv" not in os.listdir(dirpath):
            print(dirpath)
            print("b")
            continue
        if not os.path.exists(os.path.join(yaml_dir, f"{os.path.basename(dirpath)}.yaml")):
            print(dirpath)
            print("r")
            continue
        
        with open(os.path.join(yaml_dir, f"{os.path.basename(dirpath)}.yaml"), 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        row_name = os.path.basename(dirpath)
        Oders_domain_paper = "yes" if "domain_order" in yaml_data["dataset_params"] else "no"
        teacher = "yes" if "teacher" in yaml_data else "no"
        student = "yes" if "student" in yaml_data else "no"
        model_being_evaluated = "Student"

        teacher_EWC = "N/A"
        teacher_EWC_lambda = "N/A"
        if teacher == "yes":
            teacher_EWC = "yes" if yaml_data['teacher']['model_params']['train_with_ewc'] else "no"
            if teacher_EWC == "yes":
                teacher_EWC_lambda = yaml_data['teacher']["ewc_params"]['lambda']

        student_EWC = "N/A"
        student_EWC_lambda = "N/A"
        if student == "yes":
            student_EWC = "yes" if yaml_data['student']['model_params']['train_with_ewc'] else "no"
            if student_EWC == "yes":
                student_EWC_lambda = yaml_data['student']["ewc_params"]['lambda']
        
        
        temperature_distillation = yaml_data["training_params"].get("temperature", "N/A")

        mean_stability_top1 = "N/A"
        mean_stability_top5 = "N/A"
        plasticity_top1 = "N/A"
        plasticity_top5 = "N/A"

        if "afterEachdomain_top1_plasticity.csv" in os.listdir(dirpath):
            with open(os.path.join(dirpath, "afterEachdomain_top1_plasticity.csv"), 'r') as f:
                data = f.read().splitlines()
                plasticity = 0
                for idx in range(1, len(data)-1):
                    plasticity += float(data[idx].split(",")[idx])
                plasticity_top1 = plasticity/len(data)
                stability_line = list(map(float, data[-1].split(",")[1:]))
                mean_stability_top1 = sum(stability_line)/len(stability_line)

        if "afterEachdomain_top5_plasticity.csv" in os.listdir(dirpath):
            with open(os.path.join(dirpath, "afterEachdomain_top5_plasticity.csv"), 'r') as f:
                data = f.read().splitlines()
                plasticity = 0
                for idx in range(1, len(data)-1):
                    plasticity += float(data[idx].split(",")[idx])
                plasticity_top5 = plasticity/len(data)
                stability_line = list(map(float, data[-1].split(",")[1:]))
                mean_stability_top5 = sum(stability_line)/len(stability_line)
        
        results.append([row_name, 
                        Oders_domain_paper, 
                        teacher, 
                        student, 
                        teacher_EWC, 
                        student_EWC, 
                        teacher_EWC_lambda,
                          student_EWC_lambda,
                            temperature_distillation,
                              model_being_evaluated, 
                              mean_stability_top1,
                                plasticity_top1,
                                  mean_stability_top1, 
                                  mean_stability_top5])
    except:
        print(dirpath)
        continue

column_names = ["yaml_name", 
                "Order Domains as in the paper", 
                "Teacher", 
                "Student", 
                "EWC Teacher", 
                "EWC Student", 
                "EWC lambda Teacher", 
                "EWC lambda Student", 
                "Temperature Distillation", 
                "model_being_evaluated e.g Student or Teacher", 
                "Stability", 
                "Plasticity", 
                "Average Accuracy Top_1", 
                "Average Accuracy Top_5"]       


with open("final_results.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)
    writer.writerows(results)

df = pd.DataFrame(results, columns=column_names)
df.to_excel("final_results.xlsx", index=False)
