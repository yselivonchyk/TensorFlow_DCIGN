import numpy as np
import pandas as pd
import utils as ut
import sys
import os
import sklearn.metrics as m
import visualization as vis
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def get_evaluation(path):
  f = ut.get_latest_file(path, filter=r'.*.npy$')
  print(f)
  dict = np.load(f).item()
  return dict


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
  # print('predictive error(naive): %.9f (%.9f) -> %.2f%%'
  #       % (pred_mean, naive_mean, improvement*100))
  return improvement, pred_mean, naive_mean


def distance_binary_improvement(prediction_dist, naive_dist):
  pairwise = naive_dist - prediction_dist
  pairwise[pairwise > 0] = 1
  pairwise[pairwise < 0] = 0
  fraction = np.mean(pairwise)
  # print('Pairwise error improved for : %f%%' % (fraction*100), pairwise)
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
    # assert i in indexes or d[indexes[0]][0] == 0
    if i-1 in indexes:
      total += 1
    if i+1 in indexes:
      total += 1
    # print(i, i-1 in indexes, i+1 in indexes, indexes)
  metric = total/(len(point_array) - 2) / 2
  # print('NN metric: %.7f%%' % (metric * 100))
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
  # print('NN metric for preditcion: %.7f%%' % (metric * 100))
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


def reco_error(x, y):
  delta = x-y
  error = m.mean_squared_error(x.flatten(), y.flatten())
  return error


def print_folder_metrics(path):
  eval = get_evaluation(path)
  enc = eval['enc']
  pred = enc[1:-1]*2 - enc[0:-2]
  ref = enc[2:]

  # print(enc[0], enc[1], pred[0], enc[2])

  pred_to_target_dist, next_dist = distance(ref, pred), distance(enc[1:-1], enc[2:])
  pl2, pred_d, naiv_d = distance_improvement(pred_to_target_dist, next_dist)
  pb = distance_binary_improvement(pred_to_target_dist, next_dist)
  pnn = nn_metric(enc)
  pnnp = (nn_metric_pred(pred, ref)*100)
  lreco = reco_error(eval['rec'], eval['blu'])
  info = '%.3f & %.3f & %.3f & %.2f' % (pl2, pb, pnn, lreco)
  print(info)
  print('pimp:%f(%f/%f) & pb:%f & pnn:%f' % (pl2,  pred_d, naiv_d, pb, pnn), 'pnnp: %f' % pnnp)

  return info


def plot_single_cross_section_3d(data, select, subplot):
  data = data[:, select]
  # subplot.scatter(data[:, 0], data[:, 1], s=20, lw=0, edgecolors='none', alpha=1.0,
  # subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=1, alpha=0.4)

  d = data
  # subplot.plot(d[[-1, 0], 0], d[[-1, 0], 1], d[[-1, 0], 2], lw=1, alpha=0.8, color='red')
  # subplot.scatter(d[[-1, 0], 0], d[[-1, 0], 1], d[[-1, 0], 2], lw=10, alpha=0.3, marker=".", color='b')
  d = data
  subplot.scatter(d[:, 0], d[:, 1], d[:, 2], s=4, alpha=1.0, lw=0.5,
                  c=vis._build_radial_colors(len(d)),
                  marker=".",
                  cmap=plt.cm.hsv)
  subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=0.2, alpha=0.9)

  subplot.set_xlim([-0.01, 1.01])
  subplot.set_ylim([-0.01, 1.01])
  subplot.set_zlim([-0.01, 1.01])
  ticks = []
  subplot.xaxis.set_ticks(ticks)
  subplot.yaxis.set_ticks(ticks)
  subplot.zaxis.set_ticks(ticks)
  subplot.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
  subplot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))


def plot_single_cross_section_line(data, select, subplot):
  data = data[:, select]
  # subplot.scatter(data[:, 0], data[:, 1], s=20, lw=0, edgecolors='none', alpha=1.0,
  # subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=1, alpha=0.4)

  d = data
  # subplot.plot(d[[-1, 0], 0], d[[-1, 0], 1], d[[-1, 0], 2], lw=1, alpha=0.8, color='red')
  # subplot.scatter(d[[-1, 0], 0], d[[-1, 0], 1], d[[-1, 0], 2], lw=10, alpha=0.3, marker=".", color='b')
  d = data
  subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=1, alpha=0.4)

  subplot.set_xlim([-0.01, 1.01])
  subplot.set_ylim([-0.01, 1.01])
  subplot.set_zlim([-0.01, 1.01])
  ticks = []
  subplot.xaxis.set_ticks(ticks)
  subplot.yaxis.set_ticks(ticks)
  subplot.zaxis.set_ticks(ticks)
  subplot.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
  subplot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))


if __name__ == '__main__':
  path = os.getcwd()

  for _, paths, _ in os.walk(path):
    print('dirs', paths)
    break

  if len(paths) == 0:
    print_folder_metrics(path)

    eval = get_evaluation(path)
    enc = eval['enc']

    fig = vis.get_figure(shape=[1800, 900, 3])
    # ax =
    plot_single_cross_section_3d(enc, [0, 1, 2], plt.subplot(121, projection='3d'))
    plot_single_cross_section_line(enc, [0, 1, 2], plt.subplot(122, projection='3d'))
    plt.tight_layout()
    plt.show()
  else:
    res = []
    for d in paths + [path]:
      print(d)
      c_path = os.path.join(path, d)
      info = print_folder_metrics(c_path)
      res.append('\n%30s:\n%s' % (d, info))
    res = sorted(res)
    print('\n'.join(res))
    print(len(paths), len(res))
    exit(0)


  if 'TensorFlow_DCIGN' in os.getcwd().split('/')[-1]:
    # path = '/home/eugene/repo/TensorFlow_DCIGN/tmp/noise.f20_f4__i_grid03.14.c'
    # path = '/mnt/code/vd/TensorFlow_DCIGN/tmp/pred.f101_f3__i_romb8.5.6'
    path = '/media/eugene/back up/VD_backup/tmp_epoch20_final/pred.16c3s2_32c3s2_32c3s2_23c3_f3__i_romb8.5.6_'
    # path = '/media/eugene/back up/VD_backup/tmp_epoch19_inputs/pred.16c3s2_32c3s2_32c3s2_16c3_f100_f3__i_grid.28.gh.360'


