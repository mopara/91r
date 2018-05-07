import itertools as it
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class VAE(object):
  def encode(self, x):
    h = self.pre(x)

    return (self.mu_fc(h), self.log_sigma_fc(h))

  def sample(self, mu, log_sigma):
    if self.training:
      return t.randn_like(log_sigma).mul_((0.5*log_sigma).exp_()).add_(mu)
    else:
      return mu

  def forward(self, x, y):
    mu, log_sigma = self.encode(x)
    y_prd = self.decode(self.sample(mu, log_sigma))

    bce = f.binary_cross_entropy(y_prd, y, size_average=False)
    kld = -0.5*(1+log_sigma).sub_(mu.pow(2)).sub_(log_sigma.exp()).sum()

    return (y_prd, bce+kld)

class VAE1(VAE):
  def __init__(self, D_in, D_hidden, D_latent):
    super(VAE1, self).__init__()

    self.pre = pre = nn.Sequential(
      nn.Linear(D_in, D_hidden),
      nn.ReLU())
    self.mu_fc = mu_fc = nn.Linear(D_hidden, D_latent)
    self.log_sigma_fc = log_sigma_fc = nn.Linear(D_hidden, D_latent)
    self.decode = decode = nn.Sequential(
      nn.Linear(D_latent, D_hidden),
      nn.ReLU(),
      nn.Linear(D_hidden, D_in),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(self.pre.parameters(), mu_fc.parameters(),
      log_sigma_fc.parameters(), decode.parameters()), lr=1e-3)

#  (5537, 240, 320, 3)

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

class Unflatten(nn.Module):
  def __init__(self, *shape):
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)

class VAE2(nn.Module, VAE):
  def __init__(self, C_in, C1, C2, D_latent):
    super(VAE2, self).__init__()

    self.pre = pre = nn.Sequential(
      nn.Conv2d(C_in, C1, 3, padding=1), # P' = (F-1)/2 = (3-1)/2 = 1
      nn.ReLU(),
      nn.MaxPool2d(2, padding=0), # 240, 320 -> 120, 160
      nn.Conv2d(C1, 1, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, padding=0), # 120, 160 -> 60, 80
      Flatten())
    self.mu_fc = mu_fc = nn.Linear(60*80, D_latent)
    self.log_sigma_fc = log_sigma_fc = nn.Linear(60*80, D_latent)
    self.decode = decode = nn.Sequential(
      nn.Linear(D_latent, 60*80),
      Unflatten(1, 60, 80),
      nn.Conv2d(1, C2, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2, mode="nearest"), # 60, 80 -> 120, 160
      nn.Conv2d(C2, C1, 3, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2, mode="nearest") # 120, 160 -> 240, 320
      nn.Conv2d(C1, C_in, 3, padding=1),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(self.pre.parameters(), mu_fc.parameters(),
      log_sigma_fc.parameters(), decode.parameters()), lr=1e-3)
