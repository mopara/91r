import argparse
import fm
import models
import numpy as np
import os.path as path
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

def train(model, train_batches, epochs, test_batches):
  train_hist = np.empty((epochs, 2))
  test_hist = np.empty((epochs, 2))

  N_train = len(train_batches.dataset)

  for epoch in xrange(epochs):
    begin = time.time()
    epoch_loss = 0

    model.train()

    for (x, y) in train_batches:
      batch_size = x.size(0)

      y_prd, batch_loss = model(x, y)
      epoch_loss += batch_loss.item()

      model.opt.zero_grad()
      batch_loss.backward()
      model.opt.step()

    avg_epoch_loss = epoch_loss/N_train

    train_hist[epoch,0] = epoch_loss
    train_hist[epoch,1] = avg_epoch_loss

    print "Epoch: %03d [%0.1fs]\tAverage Train Loss: %g" % (epoch,
      time.time()-begin, avg_epoch_loss)

    if test_batches:
      model.eval()

      with t.no_grad():
        loss = sum(model(x, y)[1].item() for (x, y) in test_batches)
        avg_loss = loss/len(test_batches.dataset)

      test_hist[epoch,0] = loss
      test_hist[epoch,1] = avg_loss

      print "\t\t\tAverage Test Loss: %g" % avg_loss

  if test_batches:
    return (train_hist, test_hist)
  else:
    return (train_hist, None)

# todo: dont load frames into gpu memory
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
  prefix = "/home/ra_login/91r"

  gpu = t.device("cuda")

  files = (
    ("mnist/train-images-idx3-ubyte.pt", "mnist/t10k-images-idx3-ubyte.pt"),
    # ("vids/a4.sax.pt", None),
    ("vids/e5.practice.pt", None))

  for (train_file, test_file) in files:
    train_file = path.join(prefix, train_file)

    if test_file:
      test_file = path.join(prefix, test_file)

    TRAIN_X = get_data(train_file, gpu)

    N, H, W, C = TRAIN_X.size()
    D = H * W * C

    if test_file:
      TEST_X = get_data(test_file, gpu)
    else:
      TEST_X = None

    for D_z in (2, 4, 8, 16):
      for (name, vae) in {
        "vae": models.VAE(D, 400, D_z),
        "vae2": models.VAE2(D, 128, 64, 32, D_z),
        # "cvae": models.CVAE(H, W, C, 64, 128, D_z),
        "cvae": models.CVAE(H, W, C, 8, 32, D_z),
        "infovae": models.InfoVAE(H, W, C, 64, 128, 1024, D_z)}.items():

        print ">>>>>>>>>"
        print train_file, test_file
        print name, D_z
        print "<<<<<<<<<"

        vae = vae.to(gpu)

        train_x = vae.preprocess(TRAIN_X)
        train_batches = get_batches(train_x, train_x, 128, True)

        if test_file:
          test_x = vae.preprocess(TEST_X)
          test_batches = get_batches(test_x, test_x, 128, True)
        else:
          test_x = None
          test_batches = None

        train_hist, test_hist = train(vae, train_batches, 1, test_batches)

        # ex. src/hist/vae-hist-e5.practice.pt
        train_filename = path.join(prefix, "src/hist", name + ("-D_z=%d-hist-"
          % D_z) + fm.stem(train_file))
        np.save(train_filename, train_hist)


        if test_file:
          test_filename = path.join(prefix, "src/hist", name + ("-D_z=%d-hist-"
            % D_z) + fm.stem(test_file))

          np.save(test_filename, test_hist)

        t.save(vae.state_dict(), train_filename.replace("-hist-", "-model-") +
          ".pt")

        del vae

    del train_x
    del test_x
