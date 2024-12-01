import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2

from enum import Enum
from sklearn.model_selection import train_test_split

from src.Loaders.ClassNames import domainnet_classnames


"""
TODO:
- Esollir el primer domini que sutilitza (Podria ser el real?)
"""
class partition(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class DomainNet(Dataset):
    def __init__(self, root, partition, image_size=384, validation_size=0.15, random_state=42):
        self.root = root
        self.partition = partition

        self.idx2class = domainnet_classnames
        self.class2idx = {v: k for k, v in self.idx2class.items()}

        files = os.listdir(self.root)
        domains = [f for f in files if '.' not in f]
        self.domains = domains

        # Aqui es pot fer que comenci per un domini en concret
        self.domains.sort()
        self.Domain2Use = domains[0]
        self.DomainIDX = 0

        self.data = {domain: {"paths": [], "labels": []} for domain in domains}

        if (partition == partition.TRAIN) or (partition == partition.VALIDATION):
            # Iterate over the domains and load the data
            for domain in domains:
                # Open the domain txt file and read image paths and labels
                domain_txt = os.path.join(self.root, domain + '_train.txt')
                with open(domain_txt, 'r') as f:
                    lines = f.readlines()

                paths = [line.split(' ')[0] for line in lines]
                labels = [int(line.split(' ')[1]) for line in lines]

                # Split the data into training and validation
                X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=validation_size, random_state=random_state)
                
                if partition == partition.TRAIN:
                    self.data[domain]["paths"] = X_train
                    self.data[domain]["labels"] = y_train
                elif partition == partition.VALIDATION:
                    self.data[domain]["paths"] = X_val
                    self.data[domain]["labels"] = y_val

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


        else:
            # Iterate over the domains and load the data
            for domain in domains:
                # Open the domain txt file and read image paths and labels
                domain_txt = os.path.join(self.root, domain + '_test.txt')
                with open(domain_txt, 'r') as f:
                    lines = f.readlines()

                self.data[domain]["paths"] = [line.split(' ')[0] for line in lines]
                self.data[domain]["labels"] = [int(lines[0].split(' ')[1]) for line in lines]

            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    @property
    def num_domains(self):
        return len(self.domains)
    
    @property
    def num_classes(self):
        return len(self.idx2class)
    
    @property
    def current_domain(self):
        return self.Domain2Use

    @property
    def current_domain_idx(self):
        return self.DomainIDX

    @property
    def idx2domain(self):
        return {i: domain for i, domain in enumerate(self.domains)}
    
    @property
    def domain2idx(self):
        return {domain: i for i, domain in enumerate(self.domains)}

    def select_domain(self, domain: int):
        """
        Select a specific domain from the dataset. 

        Args:
            domain: The domain number to select.
        """
        self.Domain2Use = self.domains[domain]
        self.DomainIDX = domain

    def next_domain(self):
        """
        Select the next domain in the dataset.
        """
        self.DomainIDX += 1
        self.Domain2Use = self.domains[self.DomainIDX]

    def __len__(self):
        return len(self.data[self.Domain2Use]["labels"])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data[self.Domain2Use]["paths"][idx])
        label = self.data[self.Domain2Use]["labels"][idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, torch.tensor(label)

    
def get_loaders(path, image_size=384, batch_size=32):
    train_dataset = DomainNet(path, partition.TRAIN, image_size=image_size)
    val_dataset = DomainNet(path, partition.VALIDATION, image_size=image_size)
    test_dataset = DomainNet(path, partition.TEST, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset = DomainNet(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', partition=partition.TRAIN)
    print(len(dataset))
    print(dataset.current_domain)
    print(dataset.num_domains)
    print(dataset.num_classes)
    print(dataset[0])
    dataset.select_domain(1)
    print(dataset.current_domain)
    #print(dataset[0])
    dataset.next_domain()
    print(dataset.current_domain)
    print(dataset.current_domain_idx)
    print(dataset.idx2domain)
    print(dataset.domain2idx)
    #print(dataset[0])
