import itertools as it
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

def channels_first(x):
  return x.permute(0, 3, 1, 2) # NHWC -> NCHW

def flatten(x):
  return x.reshape(x.size(0), -1)

def sample(model, mean, log_var):
  if model.training:
    return t.randn_like(log_var).mul_(log_var.mul(0.5).exp_()).add_(mean)
  else:
    return mean

def reg_loss(y_prd, y, mean, log_var):
  bce = f.binary_cross_entropy(y_prd, y, size_average=False)
  kld = log_var.add(1).sub_(mean.pow(2)).sub_(log_var.exp()).sum().mul(-0.5)

  return bce.add_(kld)

def mmd_loss(y_prd, y, mean, log_var):
  pass

def forward(model, x, y, loss):
  mean, log_var = model.enc(x)
  y_prd = model.dec(sample(model, mean, log_var))

  return (y_prd, loss(y_prd, y, mean, log_var))

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return flatten(x)

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

class VAE(nn.Module):
  def __init__(self, D_in, D_h, D_z):
    super(VAE, self).__init__()

    mean_fc = nn.Linear(D_h, D_z)
    log_var_fc = nn.Linear(D_h, D_z)

    self.enc = enc = nn.Sequential(
      nn.Linear(D_in, D_h),
      nn.ReLU(),
      Lambda(mean_fc, log_var_fc))
    self.dec = dec = nn.Sequential(
      nn.Linear(D_z, D_h),
      nn.ReLU(),
      nn.Linear(D_h, D_in),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(enc.parameters(), mean_fc.parameters(),
      log_var_fc.parameters(), dec.parameters()), lr=1e-3)

  def preprocess(self, x):
    return flatten(x)

  def forward(self, x, y):
    return forward(self, x, y, reg_loss)

class CVAE(nn.Module):
  def __init__(self, H, W, C_in, C_h, D_h, D_z):
    super(CVAE, self).__init__()

    mean_fc = nn.Linear(D_h, D_z)
    log_var_fc = nn.Linear(D_h, D_z)

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
      nn.Linear(D_z, D_h),
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
      log_var_fc.parameters(), dec.parameters()), lr=1e-3, alpha=0.9, eps=1e-7)

  def preprocess(self, x):
    return channels_first(x)

  def forward(self, x, y):
    return forward(self, x, y, reg_loss)
