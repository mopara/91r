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
