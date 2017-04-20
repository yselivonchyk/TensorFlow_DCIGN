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


def distance(ref, pred):
  error = ref - pred
  return l2(error)


def l2(error):
  error = error ** 2
  error = np.sum(error, axis=1)
  return np.sqrt(error)


def distance_improvement(prediction_dist, naive_dist):
  pred_mean, naive_mean = np.mean(prediction_dist), np.mean(naive_dist)
  improvement = naive_mean-pred_mean if naive_mean-pred_mean > 0 else 0
  improvement = improvement / naive_mean
  print('predictive error(naive): %.9f (%.9f) -> %.2f%%'
        % (pred_mean, naive_mean, improvement*100))
  return improvement


def distance_binary_improvement(prediction_dist, naive_dist):
  pairwise = naive_dist - prediction_dist
  pairwise[pairwise > 0] = 1
  pairwise[pairwise < 0] = 0
  fraction = np.mean(pairwise)
  print('Pairwise error improved for : %f%%' % (fraction*100), pairwise)
  return fraction


def nn_metric(point_array):
  """
  For every frame how often previous frame t-1 and next frame t+1 are within top-2 nearest neighbours
  :param point_array: point array
  :return:
  """
  total = 0
  for i in range(1, len(point_array)-1):
    x = point_array - point_array[i]
    d = l2(x)
    indexes = np.argsort(d)[:3]
    assert i in indexes
    if i-1 in indexes:
      total += 1
    if i+1 in indexes:
      total += 1
    # print(i, i-1 in indexes, i+1 in indexes, indexes)
  metric = total/(len(point_array) - 2) / 2
  print('NN metric: %.7f%%' % (metric * 100))
  return metric


def nn_metric_pred(prediction, target):
  """
  For every frame how often previous frame t-1 and next frame t+1 are within top-2 nearest neighbours
  :param target: point array
  :return:
  """
  total = 0
  for i in range(len(target)):
    x = target - prediction[i]
    d = l2(x)
    index = np.argmin(d)
    # print(index, i, d)
    if i == index:
      total += 1
  metric = total/len(target)
  print('NN metric for preditcion: %.7f%%' % (metric * 100))
  return metric


def test_nn():
  enc = np.arange(0, 100).reshape((100, 1))
  # enc.transpose()
  # print(enc.shape)
  # print(nn_metric(enc))
  assert nn_metric(enc) == 1.


def test_nn_pred():
  enc = np.arange(0, 100).reshape((100, 1))
  # enc.transpose()
  # print(enc.shape)
  # print(nn_metric_pred(enc, enc+0.2))
  assert nn_metric_pred(enc, enc) == 1.


def print_folder_metrics(path):
  enc = get_encodding(path)
  pred = enc[1:-1]*2 - enc[0:-2]
  ref = enc[2:]

  # print(enc[0], enc[1], pred[0], enc[2])

  pred_to_target_dist, next_dist = distance(ref, pred), distance(enc[1:-1], enc[2:])
  print(distance_improvement(pred_to_target_dist, next_dist))
  print(distance_binary_improvement(pred_to_target_dist, next_dist))
  print('NN metric:', nn_metric(enc))
  print('NN metric:', nn_metric(enc))
  print('NN metric for prediction:', nn_metric_pred(pred, ref))


if __name__ == '__main__':
  # print(sys.argv)
  # if len(sys.argv) > 1:
  #   path = sys.argv[1]
  #   print('path', path)
  # else:
  #   path = '/home/eugene/repo/TensorFlow_DCIGN/tmp/noise.f20_f4__i_grid03.14.c'

  path = os.getcwd()
  if 'TensorFlow_DCIGN' in os.getcwd().split('/')[-1]:
    path = '/home/eugene/repo/TensorFlow_DCIGN/tmp/noise.f20_f4__i_grid03.14.c'

  print(path)
  # for p in ['/Volumes/unreliable/fire/VD_backup/tmp_epoch18_inputs/pred.16c3s2_32c3s2_32c3s2_16c3_f4__i_grid.28e.4',
  #           '/Volumes/unreliable/fire/VD_backup/tmp_epoch18_inputs/pred.16c3s2_32c3s2_32c3s2_23c3_f3__i_romb8.5.6']:
  test_nn()
  test_nn_pred()
  print_folder_metrics(path)
