import argparse
import cv2
import os.path as path
import numpy as np
import struct
import torch as T

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("action", type=str, help="either 'mnist' or 'vids'")
  parser.add_argument("-i", nargs="+", type=str, required=True,
    help="input file names", metavar="FILE")
  parser.add_argument("-o", type=str, required=True,
    help="destination directory", metavar="DST_DIR")

  args = parser.parse_args()

  return (args.action, args.o, args.i)

def save(dst_dir, file_name, arr):
  file_name = path.join(dst_dir, path.splitext(path.basename(file_name))[0])

  np.save(file_name, arr)
  T.save(T.tensor(arr), file_name+".pt")

  print "%s\tshape: %s" % (file_name, arr.shape)

def mnist(file_names, dst_dir):
  for file_name in file_names:
    with open(file_name, "rb") as file:
      # >: big-endian
      # H: ushort (2 bytes)
      # B: uchar (1 byte)
      # I: uint (4 bytes)
      zeros, dtype, dims = struct.unpack(">HBB", file.read(4))
      shape = [struct.unpack(">I", file.read(4))[0] for dim in range(dims)]

      # mnist input files known to have uint8 dtype
      arr = np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)

      save(dst_dir, file_name, arr)

def vids(file_names, dst_dir):
  for file_name in file_names:
    vid = cv2.VideoCapture(file_name)

    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_arr = np.empty((num_frames, height, width, 3), dtype=np.uint8)

    for frame_num in xrange(num_frames):
      ret, frame = vid.read()

      if ret:
        vid_arr[frame_num] = frame
      else:
        print "error in %s: frame [%d/%d]" % (file_name, frame_num, num_frames)
        break

    vid.release()

    save(dst_dir, file_name, vid_arr)

# ex:
# python src/data.py mnist -i mnist/src/*-ubyte -o mnist/
# python src/data.py vids -i vids/src/a4.sax.mov -o vids/
# python src/data.py vids -i vids/src/e5.practice.mov -o vids/
if __name__ == "__main__":
  action, dst_dir, file_names = parse_args()

  {"mnist": mnist, "vids": vids}[action](file_names, dst_dir)
