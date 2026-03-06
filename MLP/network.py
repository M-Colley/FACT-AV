import torch
from torch.nn import Module

class Model(Module):
    def __init__(self, input_size, num_classes=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
