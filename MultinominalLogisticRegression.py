import torch
import torch.nn as nn
import pickle

class MultinominalLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultinominalLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def serialize(self):
        model_data = {
            "weights": self.linear.weight.data,
            "biases": self.linear.bias.data
        }
        serialized_data = pickle.dumps(model_data)
        return serialized_data

    @classmethod
    def deserialize(cls, serialized_data, input_size, num_classes):
        model_data = pickle.loads(serialized_data)
        model = cls(input_size, num_classes)
        model.linear.weight.data = model_data["weights"]
        model.linear.bias.data = model_data["biases"]
        return model