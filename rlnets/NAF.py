import torch
from basenets.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class FCNAF(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # action dim
                 n_hiddens = [64, 64],  # hidden unit number list
                 nonlinear = F.tanh,
                 usebn = False,
                 action_active = None,
                 action_scaler = None
                 ):
        self.n_actions = n_actions
        n_outputfeats = 1 + self.n_actions + self.n_actions ** 2
        super(FCNAF, self).__init__(n_inputfeats, n_outputfeats, n_hiddens,
                                    nonlinear, usebn)
        # these two lines cant be moved.
        self.action_active = action_active
        self.action_scaler = action_scaler
        if action_scaler is not None:
            if isinstance(action_scaler, (int, long, float)):
                self.action_scaler = Variable(torch.Tensor([action_scaler]))
            else:
                self.action_scaler = Variable(torch.Tensor(action_scaler))

        self.tril_mask = Variable(torch.tril(torch.ones(
            self.n_actions, self.n_actions), diagonal=-1))
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(self.n_actions, self.n_actions))))

    def cuda(self, device=None):
        if self.action_scaler is not None:
            self.action_scaler = self.action_scaler.cuda()
        self.tril_mask = self.tril_mask.cuda()
        self.diag_mask = self.diag_mask.cuda()
        return self._apply(lambda t: t.cuda(device))

    def forward(self,x):
        x = MLP.forward(self, x)
        # x is a batch
        if x.dim() == 2:
            mu = x[:,:self.n_actions]
            if self.action_active is not None:
                if self.action_scaler is not None:
                    mu = self.action_scaler * self.action_active(mu)
                else:
                    mu = self.action_active(mu)
            V = x[:,self.n_actions:self.n_actions+1]
            Lunmasked_ = x[:,-self.n_actions ** 2:].clone()
            Lunmasked = Lunmasked_.view(-1,self.n_actions,self.n_actions)
            L = torch.mul(Lunmasked, self.tril_mask.unsqueeze(0)) + \
                torch.mul(torch.exp(Lunmasked), self.diag_mask.unsqueeze(0))
        elif x.dim() == 1:
            mu = x[:self.n_actions]
            if self.action_active is not None:
                if self.action_scaler is not None:
                    mu = self.action_scaler * self.action_active(mu)
                else:
                    mu = self.action_active(mu)
            V = x[self.n_actions]
            Lunmasked_ = x[-self.n_actions ** 2:].clone()
            Lunmasked = Lunmasked_.view( self.n_actions, self.n_actions)
            L = torch.mul(Lunmasked, self.tril_mask) + \
                torch.mul(torch.exp(Lunmasked), self.diag_mask)
        else:
            raise RuntimeError("dimenssion not matched")
        return V,mu,L
