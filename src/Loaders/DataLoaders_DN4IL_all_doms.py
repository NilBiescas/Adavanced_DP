import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
from enum import Enum
from sklearn.model_selection import train_test_split
from PIL import Image

from src.Loaders.ClassNames import dn4il_classnames
#from ClassNames import dn4il_classnames


class partition(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
from enum import Enum
from sklearn.model_selection import train_test_split
from PIL import Image

from src.Loaders.ClassNames import dn4il_classnames


class partition(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class DN4IL(Dataset):
    def __init__(self, root, root_dn4il, partition, image_size=384, validation_size=0.15, random_state=42, 
                 return2views=False, transform_type='default', domainOrder=None, all_domains=True):
        self.root = root
        self.root_dn4il = root_dn4il
        self.partition = partition
        self.return2views = return2views
        self.transform_type = transform_type
        self.image_size = image_size
        self.all_domains = all_domains
        self.random_state = random_state
        self.validation_size = validation_size

        self.idx2class = dn4il_classnames
        self.class2idx = {v: k for k, v in self.idx2class.items()}

        files = os.listdir(self.root)
        domains = [f for f in files if '.' not in f]
        domains.sort()
        if domainOrder is not None:
            if "real" in domains:
                domains.remove("real")
                domains = ["real"] + [d for d in domains if d != "real"]
            domains = domainOrder

        self.domains = domains
        print(f"Domains: {self.domains}")

        self.all_paths = []
        self.all_labels = []
        self.all_domain_labels = []

        if self.partition in [partition.TRAIN, partition.VALIDATION]:
            for domain_idx, domain in enumerate(self.domains):
                domain_txt = os.path.join(self.root_dn4il, domain + '_train.txt')
                with open(domain_txt, 'r') as f:
                    lines = f.readlines()
                paths = [line.strip().split(' ')[0] for line in lines]
                labels = [int(line.strip().split(' ')[1]) for line in lines]

                X_train, X_val, y_train, y_val = train_test_split(
                    paths, labels, test_size=self.validation_size, random_state=self.random_state, stratify=labels
                )

                if self.partition == partition.TRAIN:
                    self.all_paths.extend(X_train)
                    self.all_labels.extend(y_train)
                    self.all_domain_labels.extend([domain_idx] * len(X_train))
                else:
                    self.all_paths.extend(X_val)
                    self.all_labels.extend(y_val)
                    self.all_domain_labels.extend([domain_idx] * len(X_val))
        else:
            for domain_idx, domain in enumerate(self.domains):
                domain_txt = os.path.join(self.root_dn4il, domain + '_test.txt')
                with open(domain_txt, 'r') as f:
                    lines = f.readlines()
                paths = [line.strip().split(' ')[0] for line in lines]
                labels = [int(line.strip().split(' ')[1]) for line in lines]

                self.all_paths.extend(paths)
                self.all_labels.extend(labels)
                self.all_domain_labels.extend([domain_idx] * len(paths))

        if self.partition in [partition.TRAIN, partition.VALIDATION]:
            if self.transform_type == 'default':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.image_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif self.transform_type == 'type1':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=.5, hue=.3),
                        transforms.RandomRotation(degrees=(0, 180)),
                    ], p=0.5),
                    transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.image_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                raise ValueError(f"Unknown transform_type {self.transform_type}")
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @property
    def num_domains(self):
        return len(self.domains)
    
    @property
    def num_classes(self):
        return len(self.idx2class)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.all_paths[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.all_labels[idx]
        domain_label = self.all_domain_labels[idx]

        img_transform = self.transform(img)

        if self.return2views and self.partition == partition.TRAIN:
            view2 = self.transform(img)
            return img_transform, view2, torch.tensor(label), torch.tensor(domain_label)
        
        return img_transform, torch.tensor(label), torch.tensor(domain_label)



def get_loaders(path, path_dn4il, image_size=224, batch_size=32, config=None):
    return2views = config['dataset_params'].get('return2views', False)
    transform_type = config['dataset_params'].get('transform_type', 'default') 
    domainOrder = config['dataset_params'].get('domain_order', None)
    validation_size = config['dataset_params'].get('validation_size', 0.15)
    random_state = config['dataset_params'].get('random_state', 42)
    print(f"Domain Order: {domainOrder}")

    all_domains = True

    train_dataset = DN4IL(path, path_dn4il, partition.TRAIN, image_size=image_size,
                          return2views=return2views, transform_type=transform_type,
                          domainOrder=domainOrder, validation_size=validation_size,
                          random_state=random_state, all_domains=all_domains)

    val_dataset = DN4IL(path, path_dn4il, partition.VALIDATION, image_size=image_size,
                        return2views=return2views, transform_type=transform_type,
                        domainOrder=domainOrder, validation_size=validation_size,
                        random_state=random_state, all_domains=all_domains)

    test_dataset = DN4IL(path, path_dn4il, partition.TEST, image_size=image_size,
                         return2views=return2views, transform_type=transform_type,
                         domainOrder=domainOrder, all_domains=all_domains)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    return2views = False
    transform_type = 'default'
    dataset = DN4IL(
        root='/fhome/amlai07/Adavanced_DP/Data/domainnet',
        root_dn4il='/fhome/amlai07/Adavanced_DP/Data/DN4IL',
        partition=partition.TRAIN,
        all_domains=True,
        return2views=return2views,
        transform_type=transform_type
    )
    print(dataset.num_domains)
    print(len(dataset))