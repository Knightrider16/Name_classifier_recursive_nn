import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.input2h = nn.Linear(input_size, hidden_size)
        self.left_h = nn.Linear(hidden_size, hidden_size)
        self.right_h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, tree):
        if isinstance(tree, torch.Tensor):
            return torch.tanh(self.input2h(tree))
        else:
            left = self.forward(tree[0])
            right = self.forward(tree[1])
            combined = self.left_h(left) + self.right_h(right)
            return torch.tanh(combined)

    def classify(self, tree):
        h = self.forward(tree)
        output = self.h2o(h)
        return F.log_softmax(output, dim=1)
