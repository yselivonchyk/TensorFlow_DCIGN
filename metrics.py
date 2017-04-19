import numpy as np
import pandas as pd
import utils as ut
import sys
import os

def get_encodding(path):
  f = ut.get_latest_file(path, filter=r'.*.npy$')
  print(f)
  dict = np.load(f).item()
  return dict['enc']


def calculate_error(ref, pred):
  error = ref - pred
  error = error ** 2
  error = np.sum(error, axis=1)
  # print(error)
  error = np.mean(np.sqrt(error))
  return error


def print_folder_metrics(path):
  enc = get_encodding(path)
  print()

  pred =  enc[1:-1]*2 - enc[0:-2]
  ref = enc[2:]

  # print(enc[0], enc[1], pred[0], enc[2])

  prediciton_error = calculate_error(ref, pred)
  naive_error = calculate_error(enc[0:-1], enc[1:])
  print('predictive error(naive): %f (%f) -> %.2f%%' % (prediciton_error, naive_error,
        (naive_error-prediciton_error)/naive_error*100), enc.shape, path)


if __name__ == '__main__':
  print(sys.argv)
  if len(sys.argv) > 1:
    path = sys.argv[1]
    print('path', path)
  else:
    path = '/mnt/code/vd/TensorFlow_DCIGN/tmp/pred.f100_f3__i_romb8.5.6'

  path = os.getcwd()
  print(path)
  # for p in ['/Volumes/unreliable/fire/VD_backup/tmp_epoch18_inputs/pred.16c3s2_32c3s2_32c3s2_16c3_f4__i_grid.28e.4',
  #           '/Volumes/unreliable/fire/VD_backup/tmp_epoch18_inputs/pred.16c3s2_32c3s2_32c3s2_23c3_f3__i_romb8.5.6']:
  print_folder_metrics(path)
