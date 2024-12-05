from collections import Counter
from pprint import pprint

from src.Loaders.DataLoaders_DN4IL import DN4IL, partition

train_dataset = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL", partition=partition.TRAIN)
val_dataset = DN4IL(root='/fhome/amlai07/Adavanced_DP/Data/domainnet', root_dn4il="/fhome/amlai07/Adavanced_DP/Data/DN4IL", partition=partition.VALIDATION)

train_dist_all = Counter()
val_dist_all = Counter()

for i in range(train_dataset.num_domains):
    train_dataset.select_domain(i)
    val_dataset.select_domain(i)

    val_paths = val_dataset.data[val_dataset.current_domain]["paths"]
    train_paths = train_dataset.data[train_dataset.current_domain]["paths"]
    common_paths = set(val_paths).intersection(set(train_paths))

    print(f"Domain {i}: {train_dataset.current_domain}")
    print(f"Number of images in the validation set: {len(val_paths)}")
    print(f"Number of images in the training set: {len(train_paths)}")
    print(f"Number of common images between the validation and training set: {len(common_paths)}")

    train_dist = Counter(train_dataset.data[train_dataset.current_domain]["labels"])
    train_dist_all.update(train_dist)
    val_dist = Counter(val_dataset.data[val_dataset.current_domain]["labels"])
    val_dist_all.update(val_dist)


pprint(train_dist_all)
pprint(val_dist_all)
