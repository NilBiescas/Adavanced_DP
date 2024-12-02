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

    
if __name__ == '__main__':
    model = baseline()

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)