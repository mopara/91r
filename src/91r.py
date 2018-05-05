import argparse
import models
import numpy as np
import torch as T
import torch.utils.data as data

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
    loss = 0

    for (X, Y) in batches:
      Y_prd, batch_loss = model(X, Y)
      loss += batch_loss

      model.opt.zero_grad()
      batch_loss.backward()
      model.opt.step()

    print "Epoch: %03d\tAverage Train Loss: %g" % (epoch, loss/N)

def test(model, batches):
  model.eval()

  N = len(batches.dataset)
  loss = 0

  with T.no_grad():
    for (X, Y) in batches:
      Y_prd, batch_loss = model(X, Y)
      loss += batch_loss

  print "Average Test Loss: %g" % (loss/N)

def get_batches(X, Y):
  return data.DataLoader(data.TensorDataset(X, Y), batch_size=batch_size,
    shuffle=shuffle)

def data(file_name, device):
  X = (T.load(file_name).float()/255).to(device)
  N = X.shape[0]

  return (X, X.reshape(N, -1), X.shape[1], X.shape[2], np.prod(X.shape[1:]))

# python 91r.py -s -c -b256 -n100 --train=../mnist/train-images-idx3-ubyte.T --test=../mnist/t10k-images-idx3-ubyte.T
if __name__ == "__main__":
  args = parse_args()

  if not args.random:
    T.manual_seed(args.seed)

  device = T.device("cuda" if args.cuda else "cpu")

  X_trn, Xf_trn, height, width, D_in = data(args.train, device)

  # ae = models.AE1(D, 32, D).to(device)
  # ae = models.AE2(D, 32, D, args.l1).to(device)
  # ae = models.AE3(D, D).to(device)
  ae = models.AE4(D_in).to(device)

  train(ae, get_batches(Xf_trn, Xf_trn), args.num_epochs)

  if args.test:
    X_tst, Xf_tst, _, _, _ = data(args.test, device)

    test(ae, get_batches(Xf_tst, Xf_tst))
