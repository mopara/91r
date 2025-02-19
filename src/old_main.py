import argparse
import models
import numpy as np
import time
import torch as t
import torch.utils.data as data

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("-s", "--shuffle", action="store_true", default=False,
    help="shuffle training data (default: false)")
  parser.add_argument("-r", "--random", action="store_true", default=False,
    help="randomly set seed value (default: false)")
  parser.add_argument("-b", "--batch-size", default=128, type=int,
    help="input batch size for training (default: 128)", metavar="N")
  parser.add_argument("-n", "--epochs", default=10, type=int,
    help="number of epochs to train (default: 10)", metavar="E")
  parser.add_argument("-m", "--model", default="vae", type=str,
    help="model to train (default: 'vae')", metavar="M")
  parser.add_argument("-j", "--test", type=str, help="set testing dataset",
    metavar="X")
  parser.add_argument("-i", "--train", type=str, required=True,
    help="set training dataset", metavar="X")

  return parser.parse_args()

def train(model, batches, epochs):
  model.train()

  N = len(batches.dataset)

  for epoch in xrange(epochs):
    begin = time.time()
    epoch_loss = 0

    for (x, y) in batches:
      batch_size = x.size(0)

      y_prd, batch_loss = model(x, y)
      epoch_loss += batch_loss.item()

      model.opt.zero_grad()
      batch_loss.backward()
      model.opt.step()

    print "Epoch: %03d [%0.1fs]\tAverage Train Loss: %g" % (epoch,
      time.time()-begin, epoch_loss/N)

def test(model, batches):
  model.eval()

  with t.no_grad():
    loss = sum(model(x, y)[1].item() for (x, y) in batches)

  print "Average Test Loss: %g" % (loss/len(batches.dataset))

# dont load frames into gpu memory
def get_data(file_name, device):
  x = t.load(file_name).to(device).float().div_(255)

  # 1 channel: NHW
  if len(x.size()) == 3:
    x.unsqueeze_(3)

  return x

def get_batches(x, y, batch_size, shuffle):
  return data.DataLoader(data.TensorDataset(x, y), batch_size=batch_size,
    shuffle=shuffle)

if __name__ == "__main__":
  args = parse_args()

  if not args.random:
    t.manual_seed(0)

  device = t.device("cuda" if t.cuda.is_available() else "cpu")

  x = get_data(args.train, device)

  N, H, W, C = x.size()
  D = H * W * C

  vae = {
    "vae": models.VAE(D, 400, 20),
    "vae2": models.VAE2(D, 128, 64, 32, 2),
    # "cvae": models.CVAE(H, W, C, 64, 128, 2),
    "cvae": models.CVAE(H, W, C, 8, 32, 2),
    "infovae": models.InfoVAE(H, W, C, 64, 128, 1024, 2),
  }[args.model].to(device)

  x = vae.preprocess(x)

  train(vae, get_batches(x, x, args.batch_size, args.shuffle), args.epochs)

  if args.test:
    x = vae.preprocess(get_data(args.test, device))

    test(vae, get_batches(x, x, args.batch_size, args.shuffle))
