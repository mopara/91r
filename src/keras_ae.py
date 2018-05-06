import argparse

# import keras.activations as KA
import keras.layers as KL
# import keras.losses as KL
import keras.models as KM
# import keras.optimizers as KO

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

def AE1(D_in, D_out):
  enc = KM.Sequential()
  dec = KM.Sequential()
  ae = KM.Sequential()

  enc.add(KL.Dense(128, activation="relu", input_shape=(D_in,)))
  enc.add(KL.Dense(64, activation="relu"))
  enc.add(KL.Dense(32, activation="relu"))

  dec.add(KL.Dense(64, activation="relu", input_shape=(32,)))
  dec.add(KL.Dense(128, activation="relu"))
  dec.add(KL.Dense(D_out, activation="sigmoid"))

  ae.add(enc)
  ae.add(dec)

  ae.compile(optimizer="adadelta", loss="binary_crossentropy")
  # ae.compile(optimizer="adam", loss="binary_crossentropy")

  return ae

def AE2(H, W):
  enc = KM.Sequential()
  dec = KM.Sequential()
  ae = KM.Sequential()

  enc.add(KN.Conv2D(16, (3, 3), activation="relu", padding="same",
    input_shape=(H, W, 1)))
  enc.add(KN.MaxPooling2D((2, 2), padding="same"))
  enc.add(KN.Conv2D(8, (3, 3), activation="relu", padding="same"))
  enc.add(KN.MaxPooling2D((2, 2), padding="same"))
  enc.add(KN.Conv2D(8, (3, 3), activation="relu", padding="same"))
  enc.add(KN.MaxPooling2D((2, 2), padding="same"))

  dec.add(KN.Conv2D(8, (3, 3), activation="relu", padding="same",
    input_shape=enc.layers[-1].output_shape[1:]))
  dec.add(KN.UpSampling2D((2, 2))) # 4 -> 8
  dec.add(KN.Conv2D(8, (3, 3), activation="relu", padding="same"))
  dec.add(KN.UpSampling2D((2, 2))) # 8 -> 16
  dec.add(KN.Conv2D(16, (3, 3), activation="relu")) #  16 + 3 - 1 = 14
  dec.add(KN.UpSampling2D((2, 2))) # 14 -> 28
  dec.add(KN.Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

  ae.add(enc)
  ae.add(dec)

  ae.compile(optimizer="adadelta", loss="binary_crossentropy")

  return ae

def AE3(D_in, H1, H2):
  h = KM.Sequential()
  h.add(KL.Dense(H1, activation="relu", input_shape=(D_in,)))

  mu = KM.Sequential()
  mu.add(h)

def get_data(file_name):
  X = np.load(file_name).astype(np.float64)/255

  return (X, X.reshape((X.shape[0], -1)))

# python src/keras_ae.py -s -b256 -n50 -i mnist/train-images-idx3-ubyte.npy -j mnist/t10k-images-idx3-ubyte.npy
if __name__ == "__main__":
  args = parse_args()

  X_trn, Xf_trn = get_data(args.train)
  X_tst, Xf_tst = get_data(args.test)

  Xc_trn = X_trn[:,:,:,None]
  Xc_tst = X_tst[:,:,:,None]

  print Xc_trn.shape

  H, W = X_trn.shape[1:]
  D_in = H * W

  # ae = AE1(D_in, D_in)
  ae = AE2(H, W)

  ae.fit(Xc_trn, Xc_trn, epochs=args.num_epochs, batch_size=args.batch_size,
    shuffle=args.shuffle, validation_data=(Xc_tst, Xc_tst))
