import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2

from enum import Enum
from sklearn.model_selection import train_test_split

from src.Loaders.ClassNames import dn4il_classnames
#from ClassNames import dn4il_classnames

class partition(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

from PIL import Image

class DataAugmentationDINO():
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            normalize,
        ])

    def __call__(self, image):
        crops = []   
        image = Image.fromarray(image)
        crops.append(self.global_transfo1(image))
        
        
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):    
            crops.append(self.local_transfo(image))

        return crops


class DN4IL(Dataset):
    def __init__(self, root, root_dn4il, partition, image_size=384, validation_size=0.15, random_state=42, 
                 return2views = False, transform_type='default', domainOrder=None, buffer_size=-1, size_buffer_sample=0.25):
        self.seed = random_state
        self.root = root
        self.root_dn4il = root_dn4il
        self.partition = partition
        
        
        self.return2views = return2views
        self.transform_type = transform_type
        
        self.idx2class = dn4il_classnames
        self.class2idx = {v: k for k, v in self.idx2class.items()}

        files = os.listdir(self.root)
        domains = [f for f in files if '.' not in f]

        if domainOrder is None:
            domains.remove("real")
            domains = ["real"] + sorted(domains)
        else:
            domains = domainOrder

        self.Domain2Use = domains[0]
        self.DomainIDX = 0
        self.domains = domains
        print(f"Domains: {self.domains}")

        self.data = {domain: {"paths": [], "labels": []} for domain in domains}

        if (partition == partition.TRAIN) or (partition == partition.VALIDATION):
            # Iterate over the domains and load the data
            for domain in domains:
                # Open the domain txt file and read image paths and labels
                domain_txt = os.path.join(self.root_dn4il, domain + '_train.txt')
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

            if self.transform_type == 'default':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
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

                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                
            elif self.transform_type == 'DINO':
                if self.partition == partition.TRAIN:
                    self.transform = DataAugmentationDINO(global_crops_scale=(0.14, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8)
                else:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                        transforms.CenterCrop(image_size),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            # Iterate over the domains and load the data
            for domain in domains:
                # Open the domain txt file and read image paths and labels
                domain_txt = os.path.join(self.root_dn4il, domain + '_test.txt')
                with open(domain_txt, 'r') as f:
                    lines = f.readlines()

                self.data[domain]["paths"] = [line.split(' ')[0] for line in lines]
                self.data[domain]["labels"] = [int(line.split(' ')[1]) for line in lines]

            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


        self.buffer_size = 0
        if (partition == partition.TRAIN) and (buffer_size > 0):
            self.buffer = {domain: {"paths": [], "labels": []} for domain in domains}
            self.buffer2use = {domain: {"paths": [], "labels": []} for domain in domains}
            self.buffer_size = buffer_size
            self.size_buffer_sample = size_buffer_sample


    def gen_buffer(self, domain, smaple_size):
        """
        Generate a buffer of data for the given domain. 
        """
        paths = self.data[domain]["paths"]
        labels = self.data[domain]["labels"]

        permutation = torch.randperm(len(paths), generator=torch.Generator().manual_seed(self.seed))

        # If the sample size is bigger than the number of classes, we sample at least one image per class
        if smaple_size >= len(self.idx2class):
            class_idxs = [i for i in range(len(self.idx2class))]
            idxs = []
            # Reorder the indexes so we sample at least one image per class
            labels = torch.tensor(labels)
            labels = labels[permutation]
            for i in class_idxs:
                idx = permutation[labels == i][0]
                idxs.append(idx)
            
            instances2sample = smaple_size - len(self.idx2class)
            if instances2sample > 0:
                # remove sampled indexes
                permutation = permutation[~torch.tensor(idxs)]
                idxs += permutation[:instances2sample].tolist()
        else:
            idxs = permutation[:smaple_size].tolist()

        self.buffer[domain]["paths"] = [paths[i] for i in idxs]
        self.buffer[domain]["labels"] = [labels[i] for i in idxs]

    def subsample_buffer(self, domain, sample_size):
        """
        Subsample the buffer of the given domain.
        """
        paths = self.buffer[domain]["paths"]
        labels = self.buffer[domain]["labels"]

        idxs = torch.randperm(len(paths))[:sample_size]
        self.buffer2use[domain]["paths"] = [paths[i] for i in idxs]
        self.buffer2use[domain]["labels"] = [labels[i] for i in idxs]

    def gen_buffer2use(self, current_trained_domains):
        """
        Subsample the buffer for all previous domains. 
        This is done so it does not show the same images all the time.
        """
        for i in range(current_trained_domains+1):
            self.subsample_buffer(self.domains[i], int(self.buffer_size*self.size_buffer_sample))

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

        if self.partition == partition.TRAIN:
            if (self.buffer_size > 0) and (domain != 0):
                # Generate the buffer for the current domain if it is empty 
                # # (If we already generated the buffer we do not change it)
                if len(self.buffer[self.domains[domain-1]]["paths"]) == 0:
                    self.gen_buffer(self.domains[domain-1], self.buffer_size)
                
                self.gen_buffer2use(domain-1)


    def next_domain(self):
        """
        Select the next domain in the dataset.
        """
        self.DomainIDX += 1
        self.Domain2Use = self.domains[self.DomainIDX]

        if self.partition == partition.TRAIN:
            if (self.buffer_size > 0) and (self.DomainIDX != 0):
                # Generate the buffer for the current domain if it is empty 
                # # (If we already generated the buffer we do not change it)
                if len(self.buffer[self.domains[self.DomainIDX-1]]["paths"]) == 0:
                    self.gen_buffer(self.Domain2Use, self.buffer_size)
                
                self.gen_buffer2use(self.DomainIDX-1)

    def __len__(self):
        if (self.partition == partition.TRAIN) and (self.buffer_size > 0):
            # print("Buffer Size: ", self.buffer_size)
            buffer_samples = int(self.buffer_size*self.size_buffer_sample) * (self.DomainIDX)
            # print("Buffer Samples: ", buffer_samples)
            # print("Len: ", len(self.data[self.Domain2Use]["labels"]) + buffer_samples)
            return (len(self.data[self.Domain2Use]["labels"]) + buffer_samples)    

        return len(self.data[self.Domain2Use]["labels"])
        

    def __getitem__(self, idx):
        if (self.partition == partition.TRAIN) and (self.buffer_size >= 1) and (idx >= len(self.data[self.Domain2Use]["labels"])):
            paths = self.data[self.Domain2Use]["paths"].copy()
            labels = self.data[self.Domain2Use]["labels"].copy()
            for i in range(self.DomainIDX):
                paths += self.buffer2use[self.domains[i]]["paths"].copy()
                labels += self.buffer2use[self.domains[i]]["labels"].copy()
        else:
            paths = self.data[self.Domain2Use]["paths"]
            labels = self.data[self.Domain2Use]["labels"]

        img_path = os.path.join(self.root, paths[idx])
        label = labels[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_transform = self.transform(img)    
        
        if self.return2views and self.partition == partition.TRAIN:
            view2 = self.transform(img)
            return img_transform, view2, torch.tensor(label)
        
        return img_transform, torch.tensor(label)
    
    def sample_domains(self, sample_size):
        """
        Reduce the number of samples per domain to the given sample size.
        """
        for i in range(self.num_domains):
            self.select_domain(i)
            permutation = torch.randperm(len(self.data[self.Domain2Use]["paths"]), generator=torch.Generator().manual_seed(self.seed))
            idxs = permutation[:sample_size].tolist()
            self.data[self.Domain2Use]["paths"] = [self.data[self.Domain2Use]["paths"][i] for i in idxs]
            self.data[self.Domain2Use]["labels"] = [self.data[self.Domain2Use]["labels"][i] for i in idxs]
            
            
def get_loaders(path, path_dn4il, image_size=224, batch_size=32, config=None):
    # Aixo es pel dino
    return2views = config['dataset_params'].get('return2views', False)
    transform_type = config['dataset_params'].get('transform_type', 'default') 
    domainOrder = config['dataset_params'].get('domain_order', None)
    buffer_size = config['dataset_params'].get('buffer_size', -1)
    size_buffer_sample = config['dataset_params'].get('size_buffer_sample', None)
    print(f"Domain Order: {domainOrder}")
    
    train_dataset = DN4IL(path, path_dn4il, partition.TRAIN, image_size=image_size, 
                          return2views = return2views, transform_type=transform_type,
                          domainOrder=domainOrder, buffer_size=buffer_size, size_buffer_sample=size_buffer_sample)
    
    val_dataset = DN4IL(path, path_dn4il, partition.VALIDATION, image_size=image_size, 
                        return2views = return2views, transform_type=transform_type,
                        domainOrder=domainOrder)
    
    test_dataset = DN4IL(path, path_dn4il, partition.TEST, image_size=image_size,
                         return2views = return2views, transform_type=transform_type,
                         domainOrder=domainOrder)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

from torchvision.utils import save_image

if __name__ == '__main__':
    return2views = False
    transform_type = 'default'
    dataset_buff = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL", 
                    partition=partition.TRAIN, return2views = return2views, transform_type=transform_type, buffer_size=100, size_buffer_sample=0.4)
    dataset = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL",
                    partition=partition.TRAIN, return2views = return2views, transform_type=transform_type)

    print(dataset.current_domain)
    print(dataset.num_domains)
    print(dataset.num_classes)
    print(len(dataset))

    datalodaer_normal = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    datalodaer_buffer = torch.utils.data.DataLoader(dataset_buff, batch_size=32, shuffle=True)
    for i in range(dataset.num_domains):
        dataset.select_domain(i)
        dataset_buff.select_domain(i)
        #print("Len Dataset: ", len(dataset))
        #print("Len Dataset with Buffer: ", len(dataset_buff))
        r = next(iter(datalodaer_normal))
        print("Shape normal buffer", r[0].shape)
        #s = next(iter(datalodaer_buffer))
        #print(s[0].shape)

        

    # from torch.utils.data import DataLoader
    
    # loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # s = next(iter(loader))
    # l = 0
    
    # for it in range(dataset.num_domains):
    #    dataset.select_domain(it)
    #    s = dataset[0]
    #    for r, i in enumerate(s[0]):
    #        save_image(i, f"domain_{it}_test_{r}.png")

