import argparse

import keras.activations as KA
import keras.layers as KN
import keras.losses as KL
import keras.models as KM
import keras.optimizers as KO

import numpy as np

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

# def AE():
#   input_img = Input(shape=(784,))
#   encoded = Dense(128, activation='relu')(input_img)
#   encoded = Dense(64, activation='relu')(encoded)
#   encoded = Dense(32, activation='relu')(encoded)

#   decoded = Dense(64, activation='relu')(encoded)
#   decoded = Dense(128, activation='relu')(decoded)
#   decoded = Dense(784, activation='sigmoid')(decoded)

#   model = Model(input_img, decoded)

#   model.compile(optimizer='adadelta', loss='binary_crossentropy')

  # return model

def AE():
  enc = KM.Sequential()
  dec = KM.Sequential()
  ae = KM.Sequential()

  enc.add(KN.Dense(128, activation=KA.relu))
  enc.add(KN.Dense(64, activation=KA.relu))
  enc.add(KN.Dense(32, activation=KA.relu))

  dec.add(KN.Dense(64, activation=KA.relu))
  dec.add(KN.Dense(128, activation=KA.relu))
  dec.add(KN.Dense(784, activation=KA.sigmoid))

  ae.add(enc)
  ae.add(dec)

  ae.compile(optimizer=KO.Adadelta, loss=KL.binary_crossentropy)

  return ae

def get_data(file_name):
  X = np.load(file_name).astype(np.float64)/255

  return (X, X.reshape((X.shape[0], -1)))

# python src/keras_ae.py -s -b256 -n50 -i mnist/train-images-idx3-ubyte.npy -j mnist/t10k-images-idx3-ubyte.npy
if __name__ == "__main__":
  args = parse_args()

  x_trn, xf_trn = get_data(args.train)
  x_tst, xf_tst = get_data(args.test)

  ae = AE()
  ae.fit(xf_trn, xf_trn, epochs=args.num_epochs, batch_size=args.batch_size,
    shuffle=args.shuffle, validation_data=(xf_tst, xf_tst))
