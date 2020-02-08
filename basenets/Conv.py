import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Conv(nn.Module):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_outputfeats,   # output dim
                 k_sizes,
                 channels,
                 strides,
                 fcs,  # hidden unit number list
                 nonlinear = F.relu,
                 usebn = False,
                 outactive = None,
                 outscaler = None,
                 initializer = "normal",
                 initializer_param = {}, # only for orthogonal initializer
                 ):
        super(Conv,self).__init__()
        assert len(k_sizes) == len(strides) == len(channels)
        # do some initializations
        self.nonlinear = nonlinear
        self.outactive = outactive
        if outscaler is not None:
            if isinstance(outscaler, (int, float)):
                self.outscaler = Variable(torch.Tensor([outscaler]))
            else:
                self.outscaler = Variable(torch.Tensor(outscaler))
        else:
            self.outscaler = None
        if isinstance(n_inputfeats, dict):
            h, w, c = n_inputfeats['n_s']
            g = n_inputfeats['n_g']
            self.g_c = True
        else:
            h, w, c = n_inputfeats
            g = 0
            self.g_c = False
        out_h = h
        out_w = w
        for k_size, stride in zip(k_sizes, strides):
            out_h = int((out_h - k_size) / stride + 1)
            out_w = int((out_w - k_size) / stride + 1)
        n_inputfeats = out_h * out_w * channels[-1] + g
        inlists = [n_inputfeats,] + fcs
        outlists = fcs + [n_outputfeats,]
        channels.insert(0, c)

        self.conv_layers = nn.ModuleList()
        for i in range(len(k_sizes)):
            if usebn:
                bn_layer = nn.BatchNorm2d(channels[i])
                bn_layer.weight.data.fill_(1)
                bn_layer.bias.data.fill_(0)
                self.conv_layers.append(bn_layer)
            conv_layer = nn.Conv2d(channels[i], channels[i+1], k_sizes[i], strides[i])
            nn.init.xavier_uniform_(conv_layer.weight)
            self.conv_layers.append(conv_layer)

        self.fc_layers = nn.ModuleList()
        for i, (n_inunits, n_outunits) in enumerate(zip(inlists,outlists)):
            if usebn:
                bn_layer = nn.BatchNorm1d(n_inunits)
                bn_layer.weight.data.fill_(1)
                bn_layer.bias.data.fill_(0)
                self.fc_layers.append(bn_layer)
            linear_layer = nn.Linear(n_inunits,n_outunits)
            # init network IMPORTANT!!!!!
            nn.init.constant_(linear_layer.bias, 0)
            last_layer = (i == len(outlists) - 1)
            if initializer == "normal":
                if not last_layer:
                    var = initializer_param['var'] if 'var' in initializer_param.keys() else 1./ np.sqrt(n_inunits)
                    nn.init.normal_(linear_layer.weight, 0, var)
                else:
                    var = initializer_param['last_var'] if 'last_var' in initializer_param.keys() else 1. / np.sqrt(n_inunits)
                    nn.init.normal_(linear_layer.weight, 0, var)
                print("initializing layer " + str(i + 1) + " Method: Normal. Var: " + str(var))
            elif initializer == "uniform":
                if not last_layer:
                    lower = initializer_param['lower'] if 'lower' in initializer_param.keys() else -1./ np.sqrt(n_inunits)
                    upper = initializer_param['upper'] if 'upper' in initializer_param.keys() else 1./ np.sqrt(n_inunits)
                    nn.init.uniform_(linear_layer.weight, lower, upper)
                else:
                    lower = initializer_param['last_lower'] if 'last_lower' in initializer_param.keys() else -0.01
                    upper = initializer_param['last_upper'] if 'last_upper' in initializer_param.keys() else 0.01
                    nn.init.uniform_(linear_layer.weight,lower,upper)
                print("initializing layer " + str(i + 1) + " Method: Uniform. Lower: " + str(lower) + ". Upper: " + str(upper))
            elif initializer == "orthogonal":
                if not last_layer:
                    gain = initializer_param['gain'] if 'gain' in initializer_param.keys() else np.sqrt(2)
                    nn.init.orthogonal_(linear_layer.weight, gain)
                    print("initializing layer " + str(i + 1) + " Method: Orthogonal. Gain: " + str(gain))
                else:
                    gain = initializer_param['last_gain'] if 'last_gain' in initializer_param.keys() else 0.1
                    nn.init.orthogonal_(linear_layer.weight, gain)
                print("initializing layer " + str(i + 1) + " Method: Orthogonal. Gain: " + str(gain))
            elif initializer == "xavier":
                gain = initializer_param['gain'] if 'gain' in initializer_param.keys() else 1
                nn.init.xavier_uniform_(linear_layer.weight, gain)
                print("initializing layer " + str(i + 1) + " Method: Xavier. Gain: " + str(gain))
            elif initializer == "kaiming":
                a = initializer_param['a'] if 'a' in initializer_param.keys() else 0
                nn.init.kaiming_normal_(linear_layer.weight, a)
                print("initializing layer " + str(i + 1) + " Method: Kaiming_normal. a: " + str(a))
            else:
                assert 0, "please specify one initializer."
            self.fc_layers.append(linear_layer)

    def forward(self,x, other_data = None):
        assert other_data is None or isinstance(other_data, torch.Tensor)
        assert (other_data is not None and self.g_c) or not self.g_c
        if self.g_c:
            g = other_data
        input_dim = x.dim()
        if input_dim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)

        for layernum, layer in enumerate(self.conv_layers):
            x = self.nonlinear(layer(x))

        x = x.reshape(x.size(0), -1)
        if self.g_c:
            x = torch.cat((x, g), dim = 1)
        for layernum, layer in enumerate(self.fc_layers):
            x = layer(x)
            # the last layer
            if layernum == len(self.fc_layers) -1 :
                if self.outactive is not None:
                    if self.outscaler is not None:
                        x = self.outscaler.type_as(x) * self.outactive(x)
                    else:
                        x = self.outactive(x)
            else:
                x = self.nonlinear(x)

        if input_dim == 3:
            x = x.squeeze(0)

        return x
