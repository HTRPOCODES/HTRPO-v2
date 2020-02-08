import torch
from basenets.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class FCDQN(nn.Module):
    def __init__(self,
                 n_inputfeats,  # input dim
                 n_actions,  # action dim
                 n_hiddens=[30],  # hidden unit number list
                 nonlinear=F.tanh,
                 usebn = False,
                 ):
        super(FCDQN, self).__init__()
        self.net = MLP(n_inputfeats,
                        n_actions,
                        n_hiddens = n_hiddens,
                        nonlinear = nonlinear,
                        usebn = usebn)

    def forward(self, x):
        return self.net(x)

class FCDuelingDQN(nn.Module):
    def __init__(self,
                 n_inputfeats,
                 n_actions,
                 n_hiddens=[30],
                 nonlinear=F.tanh,
                 usebn = False):
        super(FCDuelingDQN, self).__init__()
        # using MLP as hidden layers
        self.hidden_layers = MLP(
            n_inputfeats,
            n_hiddens[-1],
            n_hiddens[:-1],
            nonlinear,
            outactive = nonlinear,
            usebn = usebn
        )
        self.usebn = usebn
        if self.usebn:
            self.bn = nn.BatchNorm1d(n_hiddens[-1])
        self.V = nn.Linear(n_hiddens[-1], 1)
        self.A = nn.Linear(n_hiddens[-1], n_actions)
        self.V.weight.data.normal_(0, 0.1)
        self.A.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x = self.hidden_layers.forward(x)
        input_dim = x.dim()
        if self.usebn:
            if input_dim == 1:
                x = x.unsqueeze(0)
            x = self.bn.forward(x)
        A = self.A(x)-torch.mean(self.A(x),1,keepdim=True)
        V = self.V(x)
        if self.usebn and input_dim == 1:
            A = A.squeeze(0)
            V = V.squeeze(0)
        return A+V