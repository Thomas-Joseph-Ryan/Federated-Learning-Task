import torch
import torch.nn as nn

class MultinominalLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultinominalLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x