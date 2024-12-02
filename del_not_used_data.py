import os

data_path = "/fhome/amlai07/Adavanced_DP/Data/domainnet"

domains = os.listdir(data_path)
domains = [domain for domain in domains if not domain.endswith(".txt")]

with open("/fhome/amlai07/Adavanced_DP/Data/DN4IL/clipart_test.txt", "r") as f:
    dn4il_train = f.readlines()

with open("/fhome/amlai07/Adavanced_DP/Data/domainnet/clipart_test.txt", "r") as f:
    domainnet_train = f.readlines()

used_classes = set()
for line in dn4il_train:
    used_classes.add(line.split("/")[1])

all_classes = set()
for line in domainnet_train:
    all_classes.add(line.split("/")[1])

not_used_classes = all_classes - used_classes

for domain in domains:
    domain_path = os.path.join(data_path, domain)
    for class_ in not_used_classes:
        class_path = os.path.join(domain_path, class_)
        if os.path.exists(class_path):
            os.system(f"rm -r {class_path}")