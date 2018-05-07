import itertools as it
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class VAE1(nn.Module):
  def __init__(self, D_in, D_hidden, D_latent):
    super(VAE1, self).__init__()

    self.pre = pre = nn.Sequential(
      nn.Linear(D_in, D_hidden),
      nn.ReLU())
    self.mean_fc = mean_fc = nn.Linear(D_hidden, D_latent)
    self.log_var_fc = log_var_fc = nn.Linear(D_hidden, D_latent)
    self.decode = decode = nn.Sequential(
      nn.Linear(D_latent, D_hidden),
      nn.ReLU(),
      nn.Linear(D_hidden, D_in),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(self.pre.parameters(), mean_fc.parameters(),
      log_var_fc.parameters(), decode.parameters()), lr=1e-3)

  def encode(self, x):
    h = self.pre(x)

    return (self.mean_fc(h), self.log_var_fc(h))

  def sample(self, mean, log_var):
    if self.training:
      return t.randn_like(log_var).mul_((0.5*log_var).exp_()).add_(mean)
    else:
      return mean

  def forward(self, x, y):
    mean, log_var = self.encode(x)
    y_prd = self.decode(self.sample(mean, log_var))

    bce = f.binary_cross_entropy(y_prd, y, size_average=False)
    kld = -0.5*(1+log_var).sub_(mean.pow(2)).sub_(log_var.exp()).sum()

    return (y_prd, bce+kld)

class VAE2(nn.Module):
  def __init__(self, H, W, C_in, C1, C2, D_latent):
    super(VAE2, self).__init__()

    self.pre = pre = nn.Sequential(
      nn.Conv2d(C_in, C1, 3, padding=1), # P' = (F-1)/2 = (3-1)/2 = 1
      nn.ReLU(),
      nn.MaxPool2d(2, padding=0), # 240, 320 -> 120, 160
      nn.Conv2d(C1, 1, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, padding=0)) # 120, 160 -> 60, 80
    self.mean_fc = mean_fc = nn.Linear((H*W)/4, D_latent)
    self.log_var_fc = log_var_fc = nn.Linear((H*W)/4, D_latent)
    self.z_fc = z_fc = nn.Linear(D_latent, H*W)
    self.decode = decode = nn.Sequential(
      nn.Conv2d(1, C2, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2, mode="nearest"), # 60, 80 -> 120, 160
      nn.Conv2d(C2, C1, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2, mode="nearest"), # 120, 160 -> 240, 320
      nn.Conv2d(C1, C_in, 3, padding=1),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(self.pre.parameters(), mean_fc.parameters(),
      log_var_fc.parameters(), z_fc.parameters(), decode.parameters()),
      lr=1e-3)

  def encode(self, x):
    h = self.pre(x)
    h = h.reshape(h.size(0), -1)

    print h.shape

    return (self.mean_fc(h), self.log_var_fc(h))

  def sample(self, mean, log_var):
    if self.training:
      return t.randn_like(log_var).mul_((0.5*log_var).exp_()).add_(mean)
    else:
      return mean

  def forward(self, x, y):
    mean, log_var = self.encode(x)
    y_prd = self.decode(self.z_fc(self.sample(mean, log_var)).reshape(1, 60,
      80))

    bce = f.binary_cross_entropy(y_prd, y, size_average=False)
    kld = -0.5*(1+log_var).sub_(mean.pow(2)).sub_(log_var.exp()).sum()

    return (y_prd, bce+kld)

