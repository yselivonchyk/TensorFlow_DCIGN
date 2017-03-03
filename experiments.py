import tensorflow as tf
import numpy as np
import utils as ut
import input
import DoomModel as dm
import pickle
from datetime import datetime as dt
import sys

FLAGS = tf.app.flags.FLAGS


def search_learning_rate(lrs=[0.001, 0.0004, 0.0001, 0.00003,],
                         epochs=500):
  FLAGS.suffix = 'grid_lr'
  ut.print_info('START: search_learning_rate', color=31)

  best_result, best_args = None, None
  result_summary, result_list = [], []

  for lr in lrs:
    ut.print_info('STEP: search_learning_rate', color=31)
    FLAGS.learning_rate = lr
    model = model_class()
    meta, accuracy_by_epoch = model.train(epochs)
    result_list.append((ut.to_file_name(meta), accuracy_by_epoch))
    best_accuracy = np.min(accuracy_by_epoch)
    result_summary.append('\n\r lr:%2.5f \tq:%.2f' % (lr, best_accuracy))
    if best_result is None or best_result > best_accuracy:
      best_result = best_accuracy
      best_args = lr

  meta = {'suf': 'grid_lr_bs', 'e': epochs, 'lrs': lrs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  pickle.dump(result_list, open('search_learning_rate%d.txt' % epochs, "wb"))
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' % (best_result, best_args), 36)


def search_batch_size(bss=[50], strides=[1, 2, 5, 20], epochs=500):
  FLAGS.suffix = 'grid_bs'
  ut.print_info('START: search_batch_size', color=31)
  best_result, best_args = None, None
  result_summary, result_list = [], []

  print(bss)
  for bs in bss:
    for stride in strides:
      ut.print_info('STEP: search_batch_size %d %d' % (bs, stride), color=31)
      FLAGS.batch_size = bs
      FLAGS.stride = stride
      model = model_class()
      start = dt.now()
      # meta, accuracy_by_epoch = model.train(epochs * int(bs / bss[0]))
      meta, accuracy_by_epoch = model.train(epochs)
      meta['str'] = stride
      meta['t'] = int((dt.now() - start).seconds)
      result_list.append((ut.to_file_name(meta)[22:], accuracy_by_epoch))
      best_accuracy = np.min(accuracy_by_epoch)
      result_summary.append('\n\r bs:%d \tst:%d \tq:%.2f' % (bs, stride, best_accuracy))
      if best_result is None or best_result > best_accuracy:
        best_result = best_accuracy
        best_args = (bs, stride)

  meta = {'suf': 'grid_batch_bs', 'e': epochs, 'acu': best_result,
          'h': model.get_layer_info()}
  pickle.dump(result_list, open('search_batch_size%d.txt' % epochs, "wb"))
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))

  ut.print_info('BEST Q: %d IS ACHIEVED FOR bs, st: %d %d' % (best_result, best_args[0], best_args[1]), 36)


def search_layer_sizes(epochs=500):
  FLAGS.suffix = 'grid_h'
  ut.print_info('START: search_layer_sizes', color=31)
  best_result, best_args = None, None
  result_summary, result_list = [], []

  for _, h_encoder in enumerate([300, 700, 2500]):
    for _, h_decoder in enumerate([300, 700, 2500]):
      for _, h_narrow in enumerate([3]):
        model = model_class()
        model.layer_encoder = h_encoder
        model.layer_narrow = h_narrow
        model.layer_decoder = h_decoder
        layer_info = str(model.get_layer_info())
        ut.print_info('STEP: search_layer_sizes: ' + str(layer_info), color=31)

        meta, accuracy_by_epoch = model.train(epochs)
        result_list.append((layer_info, accuracy_by_epoch))
        best_accuracy = np.min(accuracy_by_epoch)
        result_summary.append('\n\r h:%s \tq:%.2f' % (layer_info, best_accuracy))
        if best_result is None or best_result > best_accuracy:
          best_result = best_accuracy
          best_args = layer_info

  meta = {'suf': 'grid_H_bs', 'e': epochs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  print(''.join(result_summary))
  pickle.dump(result_list, open('search_layer_sizes%d.txt' % epochs, "wb"))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR H: %s' % (best_result, best_args), 36)
  ut.plot_epoch_progress(meta, result_list)


def search_layer_sizes_follow_up():
  """train further 2 best models"""
  FLAGS.save_every = 200
  for i in range(4):
    model = model_class()
    model.layer_encoder = 500
    model.layer_narrow = 3
    model.layer_decoder = 100
    model.train(600)

    model = model_class()
    model.layer_encoder = 500
    model.layer_narrow = 12
    model.layer_decoder = 500
    model.train(600)


def print_reconstructions_along_with_originals():
  FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  model = model_class()
  files = ut.list_encodings(FLAGS.save_path)
  last_encoding = files[-1]
  print(last_encoding)
  take_only = 20
  data = np.loadtxt(last_encoding)[0:take_only]
  reconstructions = model.decode(data)
  original, _ = input.get_images(FLAGS.input_path, at_most=take_only)
  ut.print_side_by_side(original, reconstructions)


def train_couple_8_models():
  FLAGS.input_path = '../data/tmp/8_pos_delay_3/img/'

  model = model_class()
  model.set_layer_sizes([500, 5, 500])
  for i in range(10):
    model.train(1000)

  model = model_class()
  model.set_layer_sizes([1000, 10, 1000])
  for i in range(20):
    model.train(1000)


if __name__ == "__main__":
  # run function if provided as console params
  epochs = 100
  model_class = dm.DoomModel
  experiment = search_learning_rate

  if len(sys.argv) > 1:
    print(sys.argv)
    experiment = sys.argv[1]
    if experiment not in locals():
      ut.print_info('Function "%s" not found. List of available functions:' % experiment)
      ut.print_info('\n'.join([x for x in locals() if 'search' in x]))
      exit(0)
    experiment = locals()[experiment]
  if len(sys.argv) > 2:
    epochs = int(sys.argv[2])
  if len(sys.argv) > 3:
    m = __import__(sys.argv[3])
    model_class = getattr(m, sys.argv[3])

  FLAGS.suffix = 'grid'
  # FLAGS.input_path = '../data/tmp/8_pos_delay/img/'

  experiment(epochs=epochs)

  # search_layer_sizes(epochs=epochs)
  # search_batch_size(epochs=epochs)
  # FLAGS.batch_size = 40
