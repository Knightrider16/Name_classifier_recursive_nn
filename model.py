import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.input2h = nn.Linear(input_size, hidden_size)
        self.combine_h = nn.Linear(hidden_size * 2, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, tree):
        if isinstance(tree, torch.Tensor):
            return torch.tanh(self.input2h(tree))
        else:
            left = self.forward(tree[0])
            right = self.forward(tree[1])
            combined = torch.cat((left, right), dim=1)
            return torch.tanh(self.combine_h(combined))

    def classify(self, tree):
        h = self.forward(tree)
        output = self.h2o(h)
        return F.log_softmax(output, dim=1)


