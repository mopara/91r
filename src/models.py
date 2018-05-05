import itertools as it
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AE1(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(AE1, self).__init__()

    self.enc = enc = nn.Sequential(
      nn.Linear(D_in, H),
      nn.ReLU()
    )

    self.dec = dec = nn.Sequential(
      nn.Linear(H, D_out),
      nn.Sigmoid()
    )

    self.ae = ae = nn.Sequential(
      enc,
      dec
    )

    self.opt = optim.Adadelta(ae.parameters())

  def loss(self, Y_prd, Y):
    L = F.binary_cross_entropy(Y_prd, Y, reduce=False)

    # take the average over each component yi of each observation y in the
    #   batch Y_prd
    return L.mean()

  def forward(self, X):
    return self.ae(X)

class AE2(nn.Module):
  def __init__(self, D_in, H, D_out, l1):
    super(AE2, self).__init__()

    self.enc = enc = nn.Sequential(
      nn.Linear(D_in, H),
      nn.ReLU()
    )

    self.dec = dec = nn.Sequential(
      nn.Linear(H, D_out),
      nn.Sigmoid()
    )

    self.opt = optim.Adadelta(it.chain(enc.parameters(), dec.parameters()))
    self.l1 = l1

  def loss(self, Y_prd, Y, Z):
    L = F.binary_cross_entropy(Y_prd, Y)
    # call mean() bc it'll eventually by multiplied by batch_size
    L1 = self.l1 * Z.norm(p=1, dim=1).mean()

    return L + L1

  def forward(self, X):
    Z = self.enc(X)

    return (self.dec(Z), Z)

class AE3(nn.Module):
  def __init__(self, D_in, D_out):
    super(AE3, self).__init__()

    self.enc = enc = nn.Sequential(
      nn.Linear(D_in, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU())
    self.dec = dec = nn.Sequential(
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, D_out),
      nn.Sigmoid())
    self.ae = ae = nn.Sequential(enc, dec)
    self.opt = optim.Adadelta(ae.parameters())

  def loss(self, Y_prd, Y, Z):
    return F.binary_cross_entropy(Y_prd, Y)

  def forward(self, X):
    return self.ae(X)
