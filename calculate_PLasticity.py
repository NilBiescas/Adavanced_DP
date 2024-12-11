import csv
import os
import yaml
import pandas as pd

dirpath = "/fhome/amlai07/Adavanced_DP/Runs/TeacherStudent_dn4il_EWC_Teacher_only_v4_Dn4il_domain_order/afterEachdomain_top1_plasticity.csv"

with open(dirpath, 'r') as f:
    data = f.read().splitlines()
    plasticity = 0
    for idx in range(1, len(data)):
        plasticity += float(data[idx].split(",")[idx])

    plasticity_top1 = plasticity/6
    stability_line = list(map(float, data[-1].split(",")[1:]))
    mean_stability_top1 = sum(stability_line)/len(stability_line)

print("The stability at 1:", round(mean_stability_top1, 2))
print("The plasticity at 1:", round(plasticity_top1, 2))