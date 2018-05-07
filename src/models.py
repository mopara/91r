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
    self.opt = optim.Adam(it.chain(pre.parameters(), mean_fc.parameters(),
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

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.reshape(x.size(0), -1)

class Lambda(nn.Module):
  def __init__(self, f, g):
    super(Lambda, self).__init__()

    self.f = f
    self.g = g

  def forward(self, x):
    return (self.f(x), self.g(x))

class Reshape(nn.Module):
  def __init__(self, *size):
    super(Reshape, self).__init__()

    self.size = size

  def forward(self, x):
    return x.reshape(x.size(0), *self.size)

class VAE3(nn.Module):
  def __init__(self, H, W, C_in, C_h, D_h, D_latent):
    super(VAE3, self).__init__()

    mean_fc = nn.Linear(D_h, D_latent)
    log_var_fc = nn.Linear(D_h, D_latent)

    self.enc = enc = nn.Sequential(
      nn.ZeroPad2d((0, 1, 0, 1)),
      nn.Conv2d(C_in, C_h, 2),
      nn.ReLU(inplace=True),
      nn.ZeroPad2d((0, W % 2, 0, H % 2)),
      nn.Conv2d(C_h, C_h, 2, 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_h, C_h, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_h, C_h, 3, padding=1),
      nn.ReLU(inplace=True),
      Flatten(),
      nn.Linear(C_h*H/2*W/2, D_h),
      Lambda(mean_fc, log_var_fc))
    self.dec = dec = nn.Sequential(
      nn.Linear(D_latent, D_h),
      nn.ReLU(inplace=True),
      nn.Linear(D_h, C_h*H/2*W/2),
      nn.ReLU(inplace=True),
      Reshape(C_h, H/2, W/2),
      nn.ConvTranspose2d(C_h, C_h, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(C_h, C_h, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(C_h, C_h, 3, 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_h, C_in, 2),
      nn.Sigmoid())
    self.opt = optim.RMSprop(it.chain(enc.parameters(), mean_fc.parameters(),
      log_var_fc.parameters(), dec.parameters()), lr=1e-3, alpha=0.9,
      eps=1e-7)

  def sample(self, mean, log_var):
    if self.training:
      return t.randn_like(log_var).mul_((0.5*log_var).exp_()).add_(mean)
    else:
      return mean

  def forward(self, x, y):
    mean, log_var = self.enc(x)
    y_prd = self.dec(self.sample(mean, log_var))

    bce = f.binary_cross_entropy(y_prd, y, size_average=False)
    kld = -0.5*(1+log_var).sub_(mean.pow(2)).sub_(log_var.exp()).sum()

    return (y_prd, bce+kld)
