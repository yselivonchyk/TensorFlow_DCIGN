"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
import utils as ut
import input as inp
import visualization as vis
import matplotlib.pyplot as plt
import time
import sys
import getch
import model_interpreter as interpreter
import network_utils as nut
import math
from Bunch import Bunch


tf.app.flags.DEFINE_string('input_path', '../data/tmp/grid03.14.c.tar.gz', 'input folder')
tf.app.flags.DEFINE_string('input_name', '', 'input folder')
tf.app.flags.DEFINE_string('test_path', '../data/tmp/grid03.14.c.tar.gz', 'test set folder')
tf.app.flags.DEFINE_string('net', 'f19-f4', 'model configuration')
tf.app.flags.DEFINE_string('model', 'noise', 'Type of the model to use: Autoencoder (ae)'
                                               'WhatWhereAe (ww) U-netAe (u)')
tf.app.flags.DEFINE_float('alpha', 10, 'Predictive reconstruction loss weight')
tf.app.flags.DEFINE_float('beta', 5, 'Reconstruction from noisy data loss weight')
tf.app.flags.DEFINE_float('epsilon', 0.3,
                          'Diameter of epsilon sphere comparing to distance to a neighbour. <= 0.5')
tf.app.flags.DEFINE_string('comment', '', 'Comment to leave by the model')


tf.app.flags.DEFINE_float('test_max', 10000, 'max number of examples in the test set')

tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('save_every', 250, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('eval_every', 25, 'Save encoding and visualizations every')
tf.app.flags.DEFINE_integer('visualiza_max', 10, 'Max pairs to show on visualization')
tf.app.flags.DEFINE_boolean('load_state', True, 'Load state if possible ')
tf.app.flags.DEFINE_boolean('dev', False, 'Indicate development mode')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')

tf.app.flags.DEFINE_float('blur', 5.0, 'Max sigma value for Gaussian blur applied to training set')
tf.app.flags.DEFINE_boolean('new_blur', False, 'Use data augmentation as blur info')
tf.app.flags.DEFINE_integer('blur_decrease', 10000, 'Decrease image blur every X steps')

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim


def is_stopping_point(current_epoch, epochs_to_train, stop_every=None, stop_x_times=None,
                      stop_on_last=True):
  if stop_on_last and current_epoch + 1 == epochs_to_train:
    return True
  if stop_x_times is not None:
    return current_epoch % np.ceil(epochs_to_train / float(stop_x_times)) == 0
  if stop_every is not None:
    return (current_epoch + 1) % stop_every == 0


def _fetch_dataset(path, take=None):
  dataset = inp.read_ds_zip(path)  # read
  take = len(dataset) if take is None else take
  dataset = dataset[:take]
  # print(dataset.dtype, dataset.shape, np.min(dataset), np.max(dataset))
  # dataset = inp.rescale_ds(dataset, 0, 1)
  return dataset


def l2(x):
  return tf.reshape(tf.sqrt(tf.reduce_sum(x ** 2, axis=1)), (FLAGS.batch_size, 1))


def get_stats_template():
  return Bunch(
    batch=[],
    input=[],
    encoding=[],
    reconstruction=[],
    total_loss=0.,
    start=time.time())


def guard_nan(x):
  return x if not math.isnan(x) else -1.


def _blur_expand(input):
  k_size = 9
  kernels = [2, 4, 6]
  channels = [input] + [nut.blur_gaussian(input, k, k_size)[0] for k in kernels]
  res = tf.concat(channels, axis=3)
  return res


class Autoencoder:
  train_set, test_set = None, None
  permutation = None
  batch_shape = None
  epoch_size = None

  input, target = None, None    # AE placeholders
  encode, decode = None, None   # AE operations
  model = None                  # interpreted model

  encoding = None               # AE predictive evaluation placeholder
  eval_decode, eval_loss = None, None   # AE evaluation

  inputs, targets = None, None  # Noise/Predictive placeholders
  raw_inputs, raw_targets = None, None  # inputs in network-friendly representation
  models = None                 # Noise/Predictive interpreted models

  optimizer = None
  train_ae, loss_ae = None, None      # AE objective
  train_pred, loss_pred = None, None  # Predictive objective
  train_dn, loss_dn = None, None      # Denoising objective

  step = None     # operation
  step_var = None # variable


  def __init__(self, optimizer=tf.train.AdamOptimizer):
    self.optimizer_constructor = optimizer
    ut.configure_folders(FLAGS)


  # MISC


  def get_past_epochs(self):
    return int(self.step_var.eval() / self.epoch_size)

  @staticmethod
  def get_checkpoint_path():
    return os.path.join(FLAGS.save_path, '-9999.chpt')

  def get_image_shape(self):
    return self.batch_shape[2:]

  def get_decoding_shape(self):
    return self.batch_shape[:1] + self.batch_shape[2:]


  # DATA


  def fetch_datasets(self):
    self.train_set = _fetch_dataset(FLAGS.input_path)
    self.epoch_size = int(self.train_set.shape[0] / FLAGS.batch_size)
    self.batch_shape = [FLAGS.batch_size] + list(self.train_set.shape[1:])

    self.test_set = _fetch_dataset(FLAGS.test_path)
    take_test = int(FLAGS.test_max) if FLAGS.test_max > 1 else int(FLAGS.test_max * len(self.test_set))
    self.test_set = self.test_set[:take_test]

  def _batch_generator(self, x=None, y=None, shuffle=True, batches=None):
    """Returns BATCH_SIZE of couples of subsequent images"""
    x = x if x is not None else self._get_blurred_dataset()
    y = y if y is not None else x
    batches = batches if batches is not None else int(np.floor(len(x) / FLAGS.batch_size))
    self.permutation = np.arange(len(x))
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    for i in range(batches):
      batch_indexes = self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
      # batch = np.stack((dataset[batch_indexes], dataset[batch_indexes + 1], dataset[batch_indexes + 2]), axis=1)
      yield x[batch_indexes], y[batch_indexes]

  def _batch_permutation_generator(self, length, start=0, shuffle=True):
    self.permutation = np.arange(length) + start
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)
    for i in range(int(length/FLAGS.batch_size)):
      yield self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]

  _blurred_dataset, _last_blur = None, 0

  def _get_blur_sigma(self):
    calculated_sigma = FLAGS.blur - int(10 * self.step_var.eval() / FLAGS.blur_decrease) / 10.0
    return max(0, calculated_sigma)

  # @ut.timeit
  def _get_blurred_dataset(self):
    if FLAGS.blur != 0:
      current_sigma = self._get_blur_sigma()
      if current_sigma != self._last_blur:
        # print(self._last_blur, current_sigma)
        self._last_blur = current_sigma
        self._blurred_dataset = inp.apply_gaussian(self.train_set, sigma=current_sigma)
      return self._blurred_dataset if self._blurred_dataset is not None else self.train_set
    return self.train_set


  # TRAIN


  def build_ae_model(self):
    self.input = tf.placeholder(tf.uint8, self.batch_shape, name='input')
    self.target = tf.placeholder(tf.uint8, self.batch_shape, name='target')

    self.step_var = tf.Variable(0, trainable=False, name='global_step')
    self.step = tf.assign(self.step_var, self.step_var + 1)

    root = self._image_to_tensor(self.input)
    target = self._image_to_tensor(self.target)

    model = interpreter.build_autoencoder(root, FLAGS.net)
    self.encode = model.encode

    self.model = model
    self.encoding = tf.placeholder(self.encode.dtype, self.encode.get_shape(), name='encoding')
    self.eval_decode = interpreter.build_decoder(self.encoding, model.config, reuse=True)
    self.eval_loss = interpreter.l2_loss(target, self.eval_decode, name='predictive_reconstruction')

    self.loss_ae = interpreter.l2_loss(target, model.decode, name='reconstruction')
    self.optimizer = self.optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self.train_ae = self.optimizer.minimize(self.loss_ae)

    self.decode = self._tensor_to_image(model.decode)


  def build_predictive_model(self):
    self.build_ae_model()  # builds on top of AE model. Due to auxilary operations init
    self.inputs = tf.placeholder(tf.uint8, [3] + self.batch_shape, name='inputs')
    self.targets = tf.placeholder(tf.uint8, [3] + self.batch_shape, name='targets')

    # transform inputs
    self.raw_inputs = [self._image_to_tensor(self.inputs[i]) for i in range(3)]
    self.raw_targets = [self._image_to_tensor(self.targets[i]) for i in range(3)]

    # build AE objective for triplet
    config = self.model.config
    models = [interpreter.build_autoencoder(x, config) for x in self.raw_inputs]
    reco_losses = [1./3 * interpreter.l2_loss(models[i].decode, self.raw_targets[i]) for i in range(3)]  # business as usual
    self.models = models

    # build predictive objective
    pred_loss_2 = self._prediction_decode(models[1].encode*2 - models[0].encode, self.raw_targets[2])
    pred_loss_0 = self._prediction_decode(models[1].encode*2 - models[2].encode, self.raw_targets[0])

    # Stitch it all together and train
    self.loss_pred = tf.add_n(reco_losses + [pred_loss_0, pred_loss_2])
    self.train_pred = self.optimizer.minimize(self.loss_pred)

  def _prediction_decode(self, prediction, target):
    """Predict encoding t3 by encoding t2 and t1 and expect a good reconstruction"""
    # Construct graph part parts for prediction
    predict_decode = interpreter.build_decoder(prediction, self.model.config, reuse=True)
    predict_loss = 1./2 * interpreter.l2_loss(predict_decode, target, alpha=FLAGS.alpha)
    self.models += [predict_decode]
    return predict_loss


  def build_denoising_model(self):
    self.build_predictive_model()  # builds on top of predictive model. Reuses triplet encoding

    # build denoising objective
    models = self.models
    loss1 = self._noisy_decode(models[1].encode, models[0].encode)
    loss2 = self._noisy_decode(models[1].encode, models[2].encode)

    self.loss_pred = self.loss_pred + loss1 + loss2
    self.train_pred = self.optimizer.minimize(self.loss_pred)

  def _noisy_decode(self, x1, x2):
    """Distort middle encoding with [<= 1/3*dist(neigbour)] and demand good reconstruction"""
    dist = l2(x1 - x2)
    noise = dist * self.epsilon_sphere_noise()
    tf.stop_gradient(noise)
    noisy_encoding = noise + self.models[1].encode
    # tf.stop_gradient(noisy_encoding)  # or maybe here, who knows
    noisy_decode = interpreter.build_decoder(noisy_encoding, self.model.config, reuse=True)
    loss = 1./2 * interpreter.l2_loss(noisy_decode, self.raw_targets[1], alpha=FLAGS.beta)
    loss = tf.clip_by_norm(loss, clip_norm=10000)
    self.models += [noisy_decode]
    return loss

  def epsilon_sphere_noise(self):
    mv_guassian = tf.random_normal(self.model.encode.get_shape().as_list())
    norm = l2(mv_guassian)
    sphere_noise = mv_guassian / norm
    unit_noise = tf.random_uniform((FLAGS.batch_size, 1), maxval=1)
    ball_noise = sphere_noise * unit_noise
    return ball_noise * FLAGS.epsilon

  def _tensor_to_image(self, net):
    with tf.name_scope('to_image'):
      if FLAGS.new_blur:
        net = net[..., :self.batch_shape[-1]]
      net = tf.nn.relu(net)
      net = tf.cast(net <= 1, net.dtype) * net * 255
      net = tf.cast(net, tf.uint8)
      return net

  def _image_to_tensor(self, image):
    with tf.name_scope('args_transform'):
      net = tf.cast(image, tf.float32) / 255.
      if FLAGS.new_blur:
        net = _blur_expand(net)
        FLAGS.blur = 0.
    return net

  def train(self):
    self.fetch_datasets()
    self.build_ae_model()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._on_training_start(sess)

      try:
        for current_epoch in range(FLAGS.max_epochs):
          start = time.time()
          for batch in self._batch_generator():
            encoding, reconstruction, loss, _, step = sess.run(
                [self.encode, self.decode, self.loss_ae, self.train_ae, self.step],
                feed_dict={self.input: batch[0], self.target: batch[1]}
            )
            self._on_batch_finish(loss, batch, encoding, reconstruction)
          self._on_epoch_finish(current_epoch, start, sess)
        self._on_training_finish()
      except KeyboardInterrupt:
        self._on_training_abort(sess)


  def train_predictive(self):
    self.fetch_datasets()
    if 'noi' in FLAGS.model:
      print('DENOISING')
      self.build_denoising_model()
    else:
      print('PREDICTIVE')
      self.build_predictive_model()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._on_training_start(sess)

      try:
        for current_epoch in range(FLAGS.max_epochs):
          start = time.time()
          ds = self._get_blurred_dataset()
          for batch_indexes in self._batch_permutation_generator(len(ds)-2):
            batch = np.stack((ds[batch_indexes], ds[batch_indexes + 1], ds[batch_indexes + 2]))
            loss, _, step = sess.run([self.loss_pred, self.train_pred, self.step],
                                     feed_dict={self.inputs: batch, self.targets: batch})
            self._on_batch_finish(loss)
          self._on_epoch_finish(current_epoch, start, sess)
        self._on_training_finish()
      except KeyboardInterrupt:
        self._on_training_abort(sess)

  def naive_evaluate(self, sess, take):
    digest = Bunch(encoded=None, reconstructed=None, source=None)
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    for batch in self._batch_generator(blurred, shuffle=False):
      encoding = self.encode.eval(feed_dict={self.input: batch[0]})
      digest.encoded = ut.concatenate(digest.encoded, encoding)

    for batch in self._batch_generator(blurred, batches=1):
      digest.source = batch[1][:take]
      digest.reconstructed = self.decode.eval(feed_dict={self.input: batch[0]})[:take]
    return digest


  def evaluate(self, sess, take):
    digest = Bunch(encoded=None, reconstructed=None, source=None,
                   loss=.0, eval_loss=.0, dumb_loss=.0)
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    for i, batch in enumerate(self._batch_generator(blurred, shuffle=False)):
      encoding = self.encode.eval(feed_dict={self.input: batch[0]})
      digest.encoded = ut.concatenate(digest.encoded, encoding)

    expected = digest.encoded[1:-1]*2 - digest.encoded[:-2]
    digest.size = len(expected)

    for p in self._batch_permutation_generator(digest.size, shuffle=False):
      digest.loss +=      self.eval_loss.eval(feed_dict={self.encoding: digest.encoded[p + 2], self.target: blurred[p + 2]})
      digest.eval_loss += self.eval_loss.eval(feed_dict={self.encoding: expected[p], self.target: blurred[p + 2]})
      digest.dumb_loss += self.loss_ae.eval(  feed_dict={self.input:    blurred[p], self.target: blurred[p + 2]})

    for batch in self._batch_generator(blurred, batches=1):
      digest.source = batch[1][:take]
      digest.reconstructed = self.decode.eval(feed_dict={self.input: batch[0]})[:take]

    digest.dumb_loss = guard_nan(digest.dumb_loss)
    digest.eval_loss = guard_nan(digest.eval_loss)
    digest.loss = guard_nan(digest.loss)

    return digest


  # TRAINING PROGRESS EVENTS


  def _on_training_start(self, sess):
    self.summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    self.saver = tf.train.Saver()

    self._restore_model(sess)

    self.epoch_stats = get_stats_template()
    self.stats = Bunch(
      epoch_accuracy=[],
      epoch_reconstructions=[],
      permutation=None
    )

    if FLAGS.dev:
      plt.ion()
      plt.show()

  def _restore_model(self, session):
    latest_checkpoint = tf.train.latest_checkpoint(self.get_checkpoint_path()[:-10])
    ut.print_info("latest checkpoint: %s" % latest_checkpoint)
    if FLAGS.load_state and latest_checkpoint is not None:
      self.saver.restore(session, latest_checkpoint)
      ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

  def _on_batch_finish(self, loss, batch=None, encoding=None, reconstruction=None):
    self.epoch_stats.total_loss += loss

    if False:
      assert batch is not None and reconstruction is not None
      original = batch[0]
      vis.plot_reconstruction(original, reconstruction, interactive=True)

  def _on_epoch_finish(self, epoch, start_time, sess):
    elapsed = time.time() - start_time

    if is_stopping_point(epoch, FLAGS.max_epochs, FLAGS.save_every):
      self.saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self.epoch_stats.total_loss / np.prod(self.batch_shape) / self.epoch_size)
    self.epoch_stats.total_loss = guard_nan(self.epoch_stats.total_loss)
    accuracy = guard_nan(accuracy)

    if is_stopping_point(epoch, FLAGS.max_epochs, FLAGS.eval_every):
      evaluation = self.evaluate(sess, take=FLAGS.visualiza_max)

      # Hack. Don't visualize ConvNets
      if len(self.encode.get_shape().as_list()) > 2:
        evaluation.encoded = np.random.randn(50, 2)
      data = {
        'enc': np.asarray(evaluation.encoded),
        'rec': np.asarray(evaluation.reconstructed),
        'blu': np.asarray(evaluation.source)
      }
      error_info = '%d(%d|%d|%d)' % (accuracy,
                                     evaluation.loss/evaluation.size,
                                     evaluation.eval_loss/evaluation.size,
                                     evaluation.dumb_loss/evaluation.size)
      meta = Bunch(suf='encodings', e='%06d' % int(self.get_past_epochs()), er=error_info)
      np.save(meta.to_file_name(folder=FLAGS.save_path), data)
      vis.plot_encoding_crosssection(
        evaluation.encoded,
        meta.to_file_name(FLAGS.save_path, 'jpg'),
        evaluation.source,
        evaluation.reconstructed,
        interactive=FLAGS.dev)


    self.stats.epoch_accuracy.append(accuracy)
    self._print_epoch_info(accuracy, epoch, FLAGS.max_epochs, elapsed)
    if epoch + 1 != FLAGS.max_epochs:
      self.epoch_stats = get_stats_template()

  def _print_epoch_info(self, accuracy, current_epoch, epochs, elapsed):
    epochs_past = self.get_past_epochs() - current_epoch
    accuracy_info = '' if accuracy is None else '| accuracy %d' % int(accuracy)
    epoch_past_info = '' if epochs_past is None else '+%d' % (epochs_past - 1)
    epoch_count = 'Epochs %2d/%d%s' % (current_epoch + 1, epochs, epoch_past_info)
    time_info = '%2dms/bt' % (elapsed / self.epoch_size * 1000)

    examples = int(np.floor(len(self.train_set) / FLAGS.batch_size))
    loss_info = 't.loss:%d' % (self.epoch_stats.total_loss * 100 / (examples * np.prod(self.batch_shape[1:])))

    info_string = ' '.join([epoch_count, accuracy_info, time_info, loss_info])
    ut.print_time(info_string, same_line=True)

  def _on_training_finish(self):
    best_acc = np.min(self.stats.epoch_accuracy)
    ut.print_time('Best Quality: %f for %s' % (best_acc, FLAGS.net))
    self.summary_writer.close()

  def _on_training_abort(self, sess):
    print('Press ENTER to save the model')
    if getch.getch() == '\n':
      print('saving')
      self.saver.save(sess, self.get_checkpoint_path())


if __name__ == '__main__':
  ut.print_flags(FLAGS)
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  FLAGS.input_name = inp.get_input_name(FLAGS.input_path)
  if len(args) <= 1:
    FLAGS.max_epochs = 50
    FLAGS.eval_every = 1
    FLAGS.blur = 0.0

  model = Autoencoder()
  if FLAGS.model == 'ae':
    model.train()
  elif 'pred' in FLAGS.model or 'noi' in FLAGS.model:
    model.train_predictive()
  else:
    print('Do-di-li-doo doo-di-li-don')
