import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from basenets.MLP import MLP
from basenets.Conv import Conv
from torch import nn

class FCPG_Gaussian(MLP):
    def __init__(self,
                 n_inputfeats,
                 n_actions,
                 sigma,
                 n_hiddens = [30],
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = None,
                 outscaler = None,
                 initializer = "orthogonal",
                 initializer_param = {"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        self.n_actions = n_actions
        super(FCPG_Gaussian, self).__init__(
            n_inputfeats,  # input dim
            n_actions,  # output dim
            n_hiddens,  # hidden unit number list
            nonlinear,
            usebn,
            outactive,
            outscaler,
            initializer,
            initializer_param=initializer_param,
        )
        self.logstd = nn.Parameter(torch.log(sigma * torch.ones(n_actions) + 1e-8))

    def forward(self,x, other_data = None):
        x = MLP.forward(self, x, other_data)
        return x, self.logstd.expand_as(x), torch.exp(self.logstd).expand_as(x)

    def cuda(self, device = None):
        self.logstd.cuda()
        return self._apply(lambda t: t.cuda(device))

class FCPG_Softmax(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens = [10],  # hidden unit number list
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = F.softmax,
                 outscaler = None,
                 initializer = "orthogonal",
                 initializer_param = {"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        self.n_actions = n_actions
        super(FCPG_Softmax, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )

class ConvPG_Softmax(Conv):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 k_sizes = [8, 4, 3],
                 channels = [8, 16, 16],
                 strides = [4, 2, 2],
                 fcs = [32, 32, 32],  # hidden unit number list
                 nonlinear = F.relu,
                 usebn = False,
                 outactive = F.softmax,
                 outscaler = None,
                 initializer="xavier",
                 initializer_param={}
                 ):
        self.n_actions = n_actions
        super(ConvPG_Softmax, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 k_sizes,
                 channels,
                 strides,
                 fcs,
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )

# TODO: support multi-layer value function in which action is concat before the final layer
class FCVALUE(MLP):
    def __init__(self,
                 n_inputfeats,
                 n_hiddens = [30],
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = None,
                 outscaler = None,
                 initializer="orthogonal",
                 initializer_param={"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        super(FCVALUE, self).__init__(
                 n_inputfeats,
                 1,
                 n_hiddens,
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )

