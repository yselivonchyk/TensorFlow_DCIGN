import model.input
import numpy as np
import model.utils as ut
import tensorflow as tf
import model.input as inp
import model.activation_functions as act
import visualization as vis
import Batcher


def minibatch_test():
  x = np.arange(0, 100, 1)

  inp = model.input.Input(x)
  for j in range(100):
    print(inp.generate_minibatch(17))


def batcher_test():
  x = np.arange(0, 100, 1)
  y = np.arange(200, 300, 1)
  bs = 7
  batcher = Batcher.Batcher(bs, x, y)
  sum = 0
  for j in range(100):
    u, v = batcher.get_batch()
    sum += np.sum(u) + np.sum(v)
    assert np.sum(v-u) == 200*bs
  assert sum/bs == np.sum(x) + np.sum(y)

def to_file_name_test():
  print(ut.to_file_name({
    'suf': 'test',
    'e': 20,
    'act': tf.nn.sigmoid,
    'opt': tf.train.GradientDescentOptimizer(learning_rate=0.001),
    'lr': 0.001,
    'init': tf.truncated_normal_initializer(stddev=0.35)
  }))
  print(ut.to_file_name({
    'suf': 'test',
    'e': 20,
    'act': tf.nn.sigmoid,
    'opt': tf.train.GradientDescentOptimizer(learning_rate=0.001),
    'lr': 0.000001,
    'init': tf.truncated_normal_initializer
  }))


def test_time():
  ut.print_time('one')
  ut.print_time('two')


def test_activation():
  print(act.sigmoid, act.sigmoid.func, act.sigmoid.max, act.sigmoid.min)


def test_ds_scale():
  ds = [-4.0, -2.0, 2.0, 4.0]
  scaled = ut.rescale_ds(ds, -2, -1)
  assert (scaled[0] - scaled[1]) * 2 == (scaled[1] - scaled[2])
  assert min(scaled) == -2
  assert max(scaled) == -1


def test_manual_pca():
  d = np.random.rand(100, 2)
  d = np.hstack((np.zeros((100, 5)), d))
  proj = vis.manual_pca(d)
  assert proj.shape[1] == 3
  assert proj[0, 0] != 0
  assert proj[0, -1] == 0
  std = np.std(proj, axis=0)
  assert std[0] > std[1]


# to_file_name_test()
# test_manual_pca()
#
# test_activation()
# test_ds_scale()
# # test_time()
# to_file_name_test()
batcher_test()
