import torch
from basenets.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np

class FCDDPG_C(nn.Module):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # action dim
                 n_hiddens = [64, 64],  # hidden unit number list
                 nonlinear = F.relu,
                 usebn = False,
                 initializer="uniform",
                 initializer_param={"last_upper":3e-3, "last_lower":3e-3}
                 ):

        assert len(n_hiddens) >= 2, "The critic has to contain at least one hidden layer."
        super(FCDDPG_C,self).__init__()
        self.n_actions = n_actions

        self.first_layer = nn.Linear(n_inputfeats, n_hiddens[0])
        # TODO: first layer initialization should be modified according to initializer.
        # initialize the first layer
        lower = initializer_param['lower'] if 'lower' in initializer_param.keys() else -1. / np.sqrt(n_inputfeats)
        upper = initializer_param['upper'] if 'upper' in initializer_param.keys() else 1. / np.sqrt(n_inputfeats)
        nn.init.uniform_(self.first_layer.weight, lower, upper)

        self.nonlinear = nonlinear
        self.net = MLP(n_hiddens[0] + self.n_actions, 1, n_hiddens[1:],
                            nonlinear, usebn, initializer=initializer,
                            initializer_param=initializer_param)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x, a):
        # TODO: support 1-d input.
        # critic only deals with mini-batch.
        x = self.nonlinear(self.first_layer.forward(x))
        x = self.net.forward(torch.cat((x, a), dim=-1))
        return x


