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

def elbo_loss(y_prd, y, mean, log_var):
  bce = f.binary_cross_entropy(y_prd, y, size_average=False)
  kld = log_var.add(1).sub_(mean.pow(2)).sub_(log_var.exp()).sum().mul_(-0.5)

  return bce.add_(kld)

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
    return forward(self, x, y, elbo_loss)

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
    return forward(self, x, y, elbo_loss)

class InfoVAE(nn.Module):
  def __init__(self, H, W, C_in, C_h1, C_h2, D_h, D_z):
    super(InfoVAE, self).__init__()

    self.enc = enc = nn.Sequential(
      nn.Conv2d(C_in, C_h1, 4, 2, padding=1), # 28 -> 14
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(C_h1, C_h2, 4, 2, padding=1), # 14 -> 7
      nn.LeakyReLU(inplace=True),
      Flatten(),
      nn.Linear(C_h2*H/4*W/4, D_h),
      nn.LeakyReLU(inplace=True),
      nn.Linear(D_h, D_z))
    self.dec = dec = nn.Sequential(
      nn.Linear(D_z, D_h),
      nn.ReLU(),
      nn.Linear(D_h, C_h2*H/4*W/4),
      nn.ReLU(),
      Reshape(C_h2, H/4, W/4),
      nn.ConvTranspose2d(C_h2, C_h1, 4, 2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(C_h1, C_in, 4, 2, padding=1),
      nn.Sigmoid())
    self.opt = optim.Adam(it.chain(enc.parameters(), dec.parameters()))

  def preprocess(self, x):
    return channels_first(x)

  def kernel(self, a, b):
    N_a, D = a.size()
    N_b = b.size(0)

    a = a.unsqueeze(1).expand(N_a, N_b, D)
    b = b.unsqueeze(0).expand(N_a, N_b, D)

    return a.sub(b).pow_(2).mean(dim=2).div_(D).mul_(-1).exp_()

  def mmd_loss(self, y_prd, y):
    y_prd_k = self.kernel(y_prd, y_prd)
    y_k = self.kernel(y, y)
    y_prd_y_k = self.kernel(y_prd, y)

    return y_prd_y_k.mul(-2).add_(y_prd_k).add_(y_k).mean()

  def forward(self, x, y):
    z = self.enc(x)
    y_prd = self.dec(z)

    mmd = self.mmd_loss(z, t.randn_like(z))
    nll = y_prd.sub(y).pow_(2).mean()

    return (y_prd, mmd.add_(nll))

class BVAE(nn.Module):
  def __init__(self):
    pass
