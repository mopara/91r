import argparse
import os.path as path
import torch as T

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("-o", default="./", type=str,
    help="destination directory", metavar="dst_dir")
  parser.add_argument("file_names", nargs="+", type=str, help="input files")

  return parser.parse_args()

def save(arr, file_name, dst_dir):
  T.save(T.tensor(arr), path.join(dst_dir,
    path.splitext(path.basename(file_name))[0]) + ".T")
