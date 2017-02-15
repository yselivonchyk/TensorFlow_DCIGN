"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os
import numpy as np
import utils as ut
import input as inp
import tools.checkpoint_utils as ch_utils
import visualization as vis
import matplotlib.pyplot as plt
import time

tf.app.flags.DEFINE_string('suffix', 'run', 'Suffix to use to distinguish models by purpose')
tf.app.flags.DEFINE_string('input_path', '../data/tmp/grid03.14.c.tar.gz', 'input folder')
tf.app.flags.DEFINE_string('test_path', '../data/tmp/grid03.14.c.tar.gz', 'test set folder')
tf.app.flags.DEFINE_float('test_max', 10000, 'max numer of exampes in the test set')
tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_string('load_from_checkpoint', None, 'Load model state from particular checkpoint')

tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('epoch_size', 100, 'Number of batches per epoch')
tf.app.flags.DEFINE_integer('test_size', 0, 'Number of test batches per epoch')
tf.app.flags.DEFINE_integer('save_every', 250, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('save_encodings_every', 5, 'Save encoding and visualizations every')
tf.app.flags.DEFINE_boolean('load_state', True, 'Load state if possible ')

tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')

tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout probability of pre-narrow units')

tf.app.flags.DEFINE_float('blur', 5.0, 'Max sigma value for Gaussian blur applied to training set')
tf.app.flags.DEFINE_integer('blur_decrease', 50000, 'Decrease image blur every X steps')

tf.app.flags.DEFINE_boolean('dev', False, 'Indicate that model is in the development mode')

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

DEV = False


def is_stopping_point(current_epoch, epochs_to_train, stop_every=None, stop_x_times=None,
                      stop_on_last=True):
  if stop_on_last and current_epoch + 1 == epochs_to_train:
    return True
  if stop_x_times is not None:
    return current_epoch % np.ceil(epochs_to_train / float(FLAGS.vis_substeps)) == 0
  if stop_every is not None:
    return (current_epoch + 1) % stop_every == 0


def get_variable(name):
  assert FLAGS.load_from_checkpoint
  var = ch_utils.load_variable(tf.train.latest_checkpoint(FLAGS.load_from_checkpoint), name)
  return var


def get_every_dataset():
  all_data = [x[0] for x in os.walk('../data/tmp_grey/') if 'img' in x[0]]
  print(all_data)
  return all_data


class Model:
  model_id = 'base'
  test_set = None

  _writer, _saver = None, None
  _dataset, _filters = None, None

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  # MODEL

  def build_model(self):
    pass

  def _build_encoder(self):
    pass

  def _build_decoder(self, weight_init=tf.truncated_normal):
    pass

  def _build_reco_loss(self, output_placeholder):
    error = self._decode - slim.flatten(output_placeholder)
    return tf.nn.l2_loss(error, name='reco_loss')

  def train(self, epochs_to_train=5):
    pass

  # META

  def get_meta(self, meta=None):
    meta = meta if meta else {}

    meta['postf'] = self.model_id
    meta['a'] = 's'
    meta['lr'] = FLAGS.learning_rate
    meta['init'] = self._weight_init
    meta['bs'] = FLAGS.batch_size
    meta['h'] = self.get_layer_info()
    meta['opt'] = self._optimizer
    meta['inp'] = inp.get_input_name(FLAGS.input_path)
    meta['do'] = FLAGS.dropout
    return meta

  def save_meta(self, meta=None):
    if meta is None:
      meta = self.get_meta()

    ut.configure_folders(FLAGS, meta)
    meta['a'] = 's'
    meta['opt'] = str(meta['opt']).split('.')[-1][:-2]
    meta['input_path'] = FLAGS.input_path
    path = os.path.join(FLAGS.save_path, 'meta.txt')
    json.dump(meta, open(path, 'w'))

  def load_meta(self, save_path):
    path = os.path.join(save_path, 'meta.txt')
    meta = json.load(open(path, 'r'))
    FLAGS.save_path = save_path
    FLAGS.batch_size = meta['bs']
    FLAGS.input_path = meta['input_path']
    FLAGS.learning_rate = meta['lr']
    FLAGS.load_state = True
    FLAGS.dropout = float(meta['do'])
    return meta

  # DATA

  _blurred_dataset, _last_blur = None, 0

  def _get_blur_sigma(self, step=None):
    step = step if step is not None else self._current_step.eval()
    calculated_sigma = FLAGS.blur - int(10 * step / FLAGS.blur_decrease) / 10.0
    return max(0, calculated_sigma)

  def _get_blurred_dataset(self):
    if FLAGS.blur != 0:
      current_sigma = self._get_blur_sigma()
      if current_sigma != self._last_blur:
        self._last_blur = current_sigma
        self._blurred_dataset = inp.apply_gaussian(self.dataset, sigma=current_sigma)
    return self._blurred_dataset if self._blurred_dataset is not None else self.dataset

  # MISC

  def get_past_epochs(self):
    return int(self._current_step.eval() / FLAGS.epoch_size)

  @staticmethod
  def get_checkpoint_path():
    return os.path.join(FLAGS.save_path, '-9999.chpt')

  # OUTPUTS
  @staticmethod
  def _get_stats_template():
    return {
      'batch': [],
      'input': [],
      'encoding': [],
      'reconstruction': [],
      'total_loss': 0,
      'start': time.time()
    }

  _epoch_stats = None
  _stats = None

  @ut.timeit
  def restore_model(self, session):
    self._saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(self.get_checkpoint_path()[:-10])
    ut.print_info("latest checkpoint: %s" % latest_checkpoint)
    if FLAGS.load_state and latest_checkpoint is not None:
      self._saver.restore(session, latest_checkpoint)
      ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

  def _register_training_start(self, sess):
    self.summary_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

    self._epoch_stats = self._get_stats_template()
    self._stats = {
      'epoch_accuracy': [],
      'epoch_reconstructions': [],
      'permutation': None
    }

    if FLAGS.dev:
      plt.ion()
      plt.show()

  # @ut.timeit
  def _register_batch(self, loss, batch=None, encoding=None, reconstruction=None, step=None):
    self._epoch_stats['total_loss'] += loss
    if FLAGS.dev:
      assert batch is not None and reconstruction is not None
      original = batch[0][:, 0]
      vis.plot_reconstruction(original, reconstruction, interactive=True)

  MAX_IMAGES = 10

  # @ut.timeit
  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    if is_stopping_point(epoch, total_epochs, FLAGS.save_every):
      self._saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self._epoch_stats['total_loss'] / np.prod(self._batch_shape) / FLAGS.epoch_size)

    if is_stopping_point(epoch, total_epochs, FLAGS.save_encodings_every):
      digest = self.evaluate(sess, take=self.MAX_IMAGES)
      data = {
        'enc': np.asarray(digest[0]),
        'rec': np.asarray(digest[1]),
        'blu': np.asarray(digest[2][:self.MAX_IMAGES])
      }

      meta = {'suf': 'encodings', 'e': '%06d' % int(self.get_past_epochs()), 'er': int(accuracy)}
      projection_file = ut.to_file_name(meta, FLAGS.save_path)
      np.save(projection_file, data)
      vis.plot_encoding_crosssection(data['enc'], FLAGS.save_path, meta, data['blu'], data['rec'])

    self._stats['epoch_accuracy'].append(accuracy)
    self.print_epoch_info(accuracy, epoch, total_epochs, elapsed)
    if epoch + 1 != total_epochs:
      self._epoch_stats = self._get_stats_template()

  @ut.timeit
  def _register_training(self):
    best_acc = np.min(self._stats['epoch_accuracy'])
    meta = self.get_meta()
    meta['acu'] = int(best_acc)
    meta['e'] = self.get_past_epochs()
    ut.print_time('Best Quality: %f for %s' % (best_acc, ut.to_file_name(meta)))
    self.summary_writer.close()
    return meta

  def print_epoch_info(self, accuracy, current_epoch, epochs, elapsed):
    epochs_past = self.get_past_epochs() - current_epoch
    accuracy_info = '' if accuracy is None else '| accuracy %d' % int(accuracy)
    epoch_past_info = '' if epochs_past is None else '+%d' % (epochs_past - 1)
    epoch_count = 'Epochs %2d/%d%s' % (current_epoch + 1, epochs, epoch_past_info)
    time_info = '%2dms/bt' % (elapsed / FLAGS.epoch_size * 1000)

    info_string = ' '.join([
      epoch_count,
      accuracy_info,
      time_info])

    ut.print_time(info_string, same_line=True)
