import itertools as it
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class VAE(nn.Module):
  def __init__(self, D_in, D_hidden, D_latent):
    super(VAE, self).__init__()

    self.pre = pre = nn.Linear(D_in, D_hidden)
    self.mu_fc = mu_fc = nn.Linear(D_hidden, D_latent)
    self.log_sigma_fc = log_sigma_fc = nn.Linear(D_hidden, D_latent)
    self.decode = decode = nn.Sequential(
      nn.Linear(D_latent, D_hidden),
      nn.Linear(D_hidden, D_in))
    self.opt = optim.Adam(it.chain(pre.parameters(), mu_fc.parameters(),
      log_sigma_fc.parameters(), decode.parameters()), lr=1e-3)

  def encode(self, x):
    h = self.pre(x)

    return (mu_fc(h), log_sigma_fc(h))

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
