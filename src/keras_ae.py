import argparse
from keras.layers import Input, Dense
from keras.models import Model
# from keras.datasets import mnist
import numpy as np
import torch as T

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

def AE():
  input_img = Input(shape=(784,))
  encoded = Dense(128, activation='relu')(input_img)
  encoded = Dense(64, activation='relu')(encoded)
  encoded = Dense(32, activation='relu')(encoded)

  decoded = Dense(64, activation='relu')(encoded)
  decoded = Dense(128, activation='relu')(decoded)
  decoded = Dense(784, activation='sigmoid')(decoded)

  model = Model(input_img, decoded)

  model.compile(optimizer='adadelta', loss='binary_crossentropy')

  return model

def get_data(file_name):
  X = T.load(file_name).numpy().astype(np.float64)/255

  return (X, X.reshape((X.shape[0], -1)))

if __name__ == "__main__":
  args = parse_args()

  x_trn = get_data(args.train)
  x_tst = get_data(args.test)

  ae = AE()
  ae.fit(x_trn, x_trn, epochs=args.num_epochs, batch_size=args.batch_size,
    shuffle=args.shuffle, validation_data=(x_tst, x_tst))


