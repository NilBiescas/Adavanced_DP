import csv
import os
import yaml
import pandas as pd

dirpath = "/fhome/amlai07/Adavanced_DP/Runs/TeacherStudent_dn4il_EWC_Dn4il_domain_order/afterEachdomain_top1_plasticity.csv"


list_paths = {"Teacher i Student, Student amb EWC": "TeacherStudent_dn4il_EWC_student_only_domain_order_importances_v3",
                
                "Normal Finetunning": "baseline_dn4il_v2_domain_order/EvaluationTeacher",
                
                #"Full training": "", No hi ha csv, se te que fer a manualment
                
                #"Task Arithmetics": "TaskArithmetics_dn4il_40",
                
                "Self Distillation sense EWC": "TeacherStudent_dn4il_domain_order/EvaluationStudent_orderItWasTrainedOn",
                
                "model amb EWC sol": "NoTeacher_studentonlyEWC_domainorder_importances_v3",
                
                "ewc als dos amb distillation": "TeacherStudent_dn4il_EWC_Dn4il_domain_order",
                
                "teacher student, teacher amb ewc": "TeacherStudent_dn4il_EWC_Teacher_only_v4_Dn4il_domain_order",
                
                "40 samples buffer": "TaskArithmetics_dn4il_40",
                
                 "150 samples buffer": "TaskArithmetics_dn4il"
    
    }


for name, dirpath in list_paths.items():
    if len(dirpath) == 0:
        continue
    dirpath = os.path.join("/fhome/amlai07/Adavanced_DP/Runs", dirpath, "afterEachdomain_top1_plasticity.csv")
    with open(dirpath, 'r') as f:
        data = f.read().splitlines()
        plasticity = 0
        for idx in range(1, len(data)):
            plasticity += float(data[idx].split(",")[idx])

        plasticity_top1 = plasticity/6
        stability_line = list(map(float, data[-1].split(",")[1:-1]))
        mean_stability_top1 = sum(stability_line)/len(stability_line)
    print(name)
    print("The stability at 1:", round(mean_stability_top1, 2))
    print("The plasticity at 1:", round(plasticity_top1, 2))
    print("\n")