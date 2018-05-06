import argparse
import models
import numpy as np
import time
import torch as T
import torch.utils.data as data

T.backends.cudnn.enabled = False

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("-s", "--shuffle", action="store_true", default=False,
    help="shuffle training data (default: false)")
  parser.add_argument("-c", "--cuda", action="store_true", default=False,
    help="enable CUDA training")
  parser.add_argument("-r", "--random", action="store_true", default=False,
    help="randomly set seed value (default: false)")
  parser.add_argument("-b", "--batch-size", default=128, type=int,
    help="input batch size for training (default: 128)", metavar="N")
  parser.add_argument("-n", "--num-epochs", default=10, type=int,
    help="number of epochs to train (default: 10)", metavar="E")
  parser.add_argument("--l1", default=0, type=float,
    help="L1 regularization coefficient (default: 0)", metavar="L1")
  parser.add_argument("-j", "--test", type=str, help="set testing dataset",
    metavar="X")
  parser.add_argument("-i", "--train", type=str, required=True,
    help="set training dataset", metavar="X")

  return parser.parse_args()

def train(model, batches, num_epochs):
  model.train()

  N = len(batches.dataset)

  for epoch in xrange(num_epochs):
    begin = time.time()
    loss = 0

    for (X, Y) in batches:
      batch_size = X.size(0)
      print X.shape

      Y_prd, batch_loss = model(X, Y)
      loss += batch_loss

      model.opt.zero_grad()
      # keras example optimizes wrt the average not the sum--not sure why
      #   maybe bc we wanna update params as if per observation not batch? idk
      (batch_loss/batch_size).backward()
      model.opt.step()


    print "Epoch: %03d\t - %01.fs - Average Train Loss: %g" % (epoch,
      time.time()-begin, loss/N)

def test(model, batches):
  model.eval()

  N = len(batches.dataset)
  loss = 0

  with T.no_grad():
    for (X, Y) in batches:
      Y_prd, batch_loss = model(X, Y)
      loss += batch_loss

  print "Average Test Loss: %g" % (loss/N)

def get_batches(X, Y, batch_size, shuffle):
  return data.DataLoader(data.TensorDataset(X, Y), batch_size=batch_size,
    shuffle=shuffle)

def get_data(file_name, device):
  X = (T.load(file_name).float()/255).to(device)

  # 1-channel
  if len(X.size()) == 3:
    X.unsqueeze_(3)

  Xf = X.reshape(X.size(0), -1)
  Xc = X.permute(0,3,1,2) # NHWC -> NCHW

  return (X, Xf, Xc)

# python 91r/src/91r.py -s -c -n100 --train=91r/mnist/train-images-idx3-ubyte.T --test=91r/mnist/t10k-images-idx3-ubyte.T
if __name__ == "__main__":
  args = parse_args()

  if not args.random:
    T.manual_seed(1)

  device = T.device("cuda" if args.cuda else "cpu")

  X_trn, Xf_trn, Xc_trn = get_data(args.train, device)

  N, H, W, C = X_trn.size()
  D = H * W * C

  # ae = models.AE1(D, 32, D).to(device)
  # ae = models.AE2(D, 32, D, args.l1).to(device)
  # ae = models.AE3(D, D).to(device)
  # ae = models.AE4(D).to(device)
  # ae = models.AE5(D, 32, D).to(device)
  ae = models.AE6(C).to(device)

  train(ae, get_batches(Xc_trn, Xc_trn, args.batch_size, args.shuffle),
    args.num_epochs)

  if args.test:
    X_tst, Xf_tst, Xc_tst = get_data(args.test, device)

    test(ae, get_batches(Xc_tst, Xc_tst, args.batch_size, args.shuffle))
