import torch
import torch.nn as nn
import torchvision.models as models

class resnet18(nn.Module):
    def __init__(self, num_classes=100, train_with_ewc=False, **kwargs):
        super(resnet18, self).__init__()

        self.model = models.resnet18(weights='DEFAULT')
        print(self.model)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        self.train_with_ewc = train_with_ewc
        self.gradient_stop = kwargs.get("gradient_stop", False)
        print(f"train_with_ewc: {train_with_ewc}")
        print(f"gradient_stop: {self.gradient_stop}")

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    model = resnet18()
    print(model.model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)