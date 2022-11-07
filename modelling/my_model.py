import torch.nn as nn

class Ontop_Modeler(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(Ontop_Modeler, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs):
        y = self.linear1(input_xs)
        y = self.tanh(y)
        y = self.linear2(y)
        return y
