import torch
import torch.nn as nn
import torchvision.models as models

class mobilenet_v3_small(nn.Module):
    def __init__(self, num_classes=100, train_with_ewc=False, **kwargs):
        super(mobilenet_v3_small, self).__init__()

        self.model = models.mobilenet_v3_small(weights='DEFAULT')
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=256, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=num_classes, bias=True),
        )

        self.train_with_ewc = train_with_ewc
        self.gradient_stop = kwargs.get("gradient_stop", False)
        print(f"train_with_ewc: {train_with_ewc}")
        print(f"gradient_stop: {self.gradient_stop}")

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    model = mobilenet_v3_small()
    print(model.model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)