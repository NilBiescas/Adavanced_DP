# Coess iguals
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from enum import Enum

class Domains(Enum):
    clipart = 0
    infograph = 1
    painting = 2
    quickdraw = 3
    real = 4
    sketch = 5

class DomainNet(Dataset):
    def __init__(self, args):
        self.args = args
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

        self.domains_ids = 0
        
    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)
        
    def __getitem__(self, idx):
        


- Caclcular metriuques
Cada cop que evaluem cridarem a:
-   Top1, Top5,  (Pasar model, i dades del domini) i aixo per cada domini, que hem vist fins ara (parametre per si volem un domini en concret i sol aquell)




# Coses uniques
- train
- eval
- args

    - ewc calculation
    


class Experiment1(metriques)


    
    
    

for ..in

    model()
    loss
    ...
    
# 1 Experiment: Baseline: Student normal finetunning
# 2 Experiment: Teacher and Student, teacher with EWC and student with EWC
# 3 Experiment: Teacher and Student, teacher without EWC and student with EWC
# 4 Experiment: Teacher and Student, teacher without EWC and student without EWC

# 5 Experiment: DANN 
# 6 Experiment: Task arithmetic 