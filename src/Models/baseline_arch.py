import torch
import torch.nn as nn
import torchvision.models as models

class baseline(nn.Module):
    def __init__(self, num_classes=345):
        super(baseline, self).__init__()

        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=640, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=640, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        return self.model(x)



class baseline_with_ewc(nn.Module):
    def __init__(self, num_classes=345, train_with_ewc=False, **kwargs):
        super(baseline_with_ewc, self).__init__()
        
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=640, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=640, out_features=num_classes, bias=True),
        )
        
        self.train_with_ewc = train_with_ewc
        self.gradient_stop = kwargs.get("gradient_stop", False)
        print(f"train_with_ewc: {train_with_ewc}")
        print(f"gradient_stop: {self.gradient_stop}")
        
    def forward(self, x): # 
        #Podriem fer return self.model(x).detach() per tal de no calcular gradients pels casos amb gradient_stop pero bueno
        return self.model(x)


class baseline_with_ewc_DomainHeads(nn.Module):
    def __init__(self, train_with_ewc=False, **kwargs):
        super(baseline_with_ewc_DomainHeads, self).__init__()
        
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Identity()
        
        self.train_with_ewc = train_with_ewc
        self.gradient_stop = kwargs.get("gradient_stop", False)
        print(f"train_with_ewc: {train_with_ewc}")
        print(f"gradient_stop: {self.gradient_stop}")
        
    def forward(self, x):
        return self.model(x)
    
    def named_parameters(self, recurse: bool = True, exclude_classifier=True):
        """
        Override the named_parameters method to optionally exclude the classifier parameters.
        
        Args:
            exclude_classifier (bool): If True, exclude parameters in the classifier head.
        
        Yields:
            Tuple[str, nn.Parameter]: Parameter name and parameter tensor.
        """
        for name, param in super().named_parameters(recurse=recurse):
            if exclude_classifier and name.startswith('model.classifier'):
                continue
            yield name, param


class TaskHead(nn.Module):
    def __init__(self, input_size: int, # number of features in the backbone's output
                 projection_size: int,  # number of neurons in the hidden layer
                 num_classes: int,      # number of output neurons
                 dropout: float=0.,     # optional dropout rate to apply
                 device="cpu"):
        super().__init__()

        # 1280
        self.fc1 = nn.Linear(input_size, projection_size)

        self.classifier = nn.Linear(projection_size, num_classes)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.silu = nn.SiLU()

        self.device = device
        self.to(device)

    def forward(self, x):
        # assume x is already unactivated feature logits,
        # e.g. from resnet backbone
        x = self.fc1(self.dropout(x))
        x = self.classifier(self.silu(x))
        return x


    
if __name__ == '__main__':
    model = baseline_with_ewc_DomainHeads()
    
    print(model(torch.randn(1, 3, 384, 384)).shape)
    
    model.model.classifier = TaskHead(223, 100, 10)

    #for name, param in model.named_parameters():
    #    print(name)
    
    
    
