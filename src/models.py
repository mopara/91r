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
    # mean() instead of sum() bc it'll eventually by multiplied by batch_size
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
    return (self.ae(X), None)

class AE4(nn.Module):
  def __init__(self, D_in):
    super(AE4, self).__init__()

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
      nn.Linear(128, D_in),
      nn.Sigmoid())
    self.ae = ae = nn.Sequential(enc, dec)
    self.opt = optim.Adadelta(ae.parameters())

  def forward(self, X, Y):
    Y_prd = self.ae(X)

    # let l(y_prd, y) be the average bce(y_prd[i], y[i])
    # sum l(y_prd, y) over observation (y_prd, y) in the batch (Y_prd, Y)
    loss = F.binary_cross_entropy(Y_prd, Y, reduce=False).mean(dim=1).sum()

    kernel_loss = 0
    act_loss = 0

    return (Y_prd, loss+kernel_loss+act_loss)

class AE5(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(AE5, self).__init__()

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

  def forward(self, X, Y):
    Y_prd = self.ae(X)
    loss = F.binary_cross_entropy(Y_prd, Y, reduce=False).mean(dim=1).sum()
    # loss = F.binary_cross_entropy(Y_prd, Y)

    return (Y_prd, loss)
