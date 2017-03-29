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
from tensorflow.contrib.tensorboard.plugins import projector
from Bunch import Bunch


tf.app.flags.DEFINE_string('input_path', '../data/tmp/grid03.14.c.tar.gz', 'input folder')
tf.app.flags.DEFINE_string('input_name', '', 'input folder')
tf.app.flags.DEFINE_string('test_path', '', 'test set folder')
tf.app.flags.DEFINE_string('net', 'f100-f3', 'model configuration')
tf.app.flags.DEFINE_string('model', 'pred', 'Type of the model to use: Autoencoder (ae)'
                                               'WhatWhereAe (ww) U-netAe (u)')

tf.app.flags.DEFINE_float('alpha', 10, 'Predictive reconstruction loss weight')
tf.app.flags.DEFINE_float('beta', 0.0005, 'Reconstruction from noisy data loss weight')
tf.app.flags.DEFINE_float('epsilon', 0.000001,
                          'Diameter of epsilon sphere comparing to distance to a neighbour. <= 0.5')
tf.app.flags.DEFINE_float('gamma', 50., 'Loss weight for large distances')
tf.app.flags.DEFINE_float('distance', 0.01, 'Maximum allowed interpoint distance')
tf.app.flags.DEFINE_float('delta', 1., 'Loss weight for stacked objective')

tf.app.flags.DEFINE_string('comment', '', 'Comment to leave by the model')

tf.app.flags.DEFINE_float('test_max', 10000, 'max number of examples in the test set')

tf.app.flags.DEFINE_integer('max_epochs', 0, 'Train for at most this number of epochs')
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


AUTOENCODER = 'ae'
PREDICTIVE = 'pred'
DENOISING = 'noise'

CHECKPOINT_NAME = '-9999.chpt'
EMB_SUFFIX = '_embedding'


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
  ut.print_info('DS fetch: %8d (%s)' % (len(dataset), path))
  return dataset


def l2(x):
  l = x.get_shape().as_list()[0]
  return tf.reshape(tf.sqrt(tf.reduce_sum(x ** 2, axis=1)), (l, 1))


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

  encoding = None                       # AE predictive evaluation placeholder
  eval_decode, eval_loss = None, None   # AE evaluation

  inputs, targets = None, None          # Noise/Predictive placeholders
  raw_inputs, raw_targets = None, None  # inputs in network-friendly representation
  models = None                         # Noise/Predictive interpreted models

  optimizer, train = None, None
  loss_ae, loss_reco, loss_pred, loss_dn = None, None, None,  None   # Objectives
  loss_total = None
  losses = []

  step = None     # operation
  step_var = None # variable

  vis_summary, vis_placeholder = None, None
  image_summaries = None


  def __init__(self, optimizer=tf.train.AdamOptimizer):
    self.optimizer_constructor = optimizer
    FLAGS.input_name = inp.get_input_name(FLAGS.input_path)
    ut.configure_folders(FLAGS)
    ut.print_flags(FLAGS)

  # MISC


  def get_past_epochs(self):
    return int(self.step.eval() / self.epoch_size)

  @staticmethod
  def get_checkpoint_path():
    return os.path.join(FLAGS.save_path, CHECKPOINT_NAME)


  # DATA


  def fetch_datasets(self):
    self.train_set = _fetch_dataset(FLAGS.input_path)
    self.epoch_size = int(self.train_set.shape[0] / FLAGS.batch_size)
    self.batch_shape = [FLAGS.batch_size] + list(self.train_set.shape[1:])

    reuse_train = FLAGS.test_path == FLAGS.input_path or FLAGS.test_path == ''
    self.test_set = self.train_set.copy() if reuse_train else _fetch_dataset(FLAGS.test_path)
    take_test = int(FLAGS.test_max) if FLAGS.test_max > 1 else int(FLAGS.test_max * len(self.test_set))
    ut.print_info('take %d from test' % take_test)
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

  def _batch_permutation_generator(self, length, start=0, shuffle=True, batches=None):
    self.permutation = np.arange(length) + start
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)
    for i in range(int(length/FLAGS.batch_size)):
      if batches is not None and i >= batches:
        break
      yield self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]

  _blurred_dataset, _last_blur = None, 0

  def _get_blur_sigma(self):
    calculated_sigma = FLAGS.blur - int(10 * self.step.eval() / FLAGS.blur_decrease) / 10.0
    return max(0, calculated_sigma)

  # @ut.timeit
  def _get_blurred_dataset(self):
    if FLAGS.blur != 0:
      current_sigma = self._get_blur_sigma()
      if current_sigma != self._last_blur:
        # print(self._last_blur, current_sigma)
        self._last_blur = current_sigma
        self._blurred_dataset = inp.apply_gaussian(self.train_set, sigma=current_sigma)
        ut.print_info('blur s:%.1f[%.1f>%.1f]' % (current_sigma, self.train_set[2, 10, 10, 0], self._blurred_dataset[2, 10, 10, 0]))
      return self._blurred_dataset if self._blurred_dataset is not None else self.train_set
    return self.train_set


# TRAIN


  def build_ae_model(self):
    self.input = tf.placeholder(tf.uint8, self.batch_shape, name='input')
    self.target = tf.placeholder(tf.uint8, self.batch_shape, name='target')
    self.step = tf.Variable(0, trainable=False, name='global_step')

    root = self._image_to_tensor(self.input)
    target = self._image_to_tensor(self.target)

    model = interpreter.build_autoencoder(root, FLAGS.net)
    self.encode = model.encode

    self.model = model
    self.encoding = tf.placeholder(self.encode.dtype, self.encode.get_shape(), name='encoding')
    eval_decode = interpreter.build_decoder(self.encoding, model.config, reuse=True)
    print(target, eval_decode)
    self.eval_loss = interpreter.l2_loss(target, eval_decode, name='predictive_reconstruction')
    self.eval_decode = self._tensor_to_image(eval_decode)

    self.loss_ae = interpreter.l2_loss(target, model.decode, name='reconstruction')
    self.decode = self._tensor_to_image(model.decode)
    self.losses = [self.loss_ae]

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
    pred_loss_2 = self._prediction_decode(models[1].encode*2 - models[0].encode, self.raw_targets[2], models[2])
    pred_loss_0 = self._prediction_decode(models[1].encode*2 - models[2].encode, self.raw_targets[0], models[0])

    # build regularized distance objective
    dist_loss1 = self._distance_loss(models[1].encode - models[0].encode)
    dist_loss2 = self._distance_loss(models[1].encode - models[2].encode)

    # Stitch it all together and train
    self.loss_reco = tf.add_n(reco_losses)
    self.loss_pred = pred_loss_0 + pred_loss_2
    self.loss_dist = dist_loss1 + dist_loss2
    self.losses = [self.loss_reco, self.loss_pred]

  def _distance_loss(self, distances):
    error = tf.nn.relu(l2(distances) - FLAGS.distance ** 2)
    return tf.reduce_sum(error)

  def _prediction_decode(self, prediction, target, model):
    """Predict encoding t3 by encoding (t2 and t1) and expect a good reconstruction"""
    predict_decode = interpreter.build_decoder(prediction, self.model.config, reuse=True, masks=model.mask_list)
    predict_loss = 1./2 * interpreter.l2_loss(predict_decode, target, alpha=FLAGS.alpha)
    self.models += [predict_decode]
    return predict_loss * FLAGS.gamma


  def build_denoising_model(self):
    self.build_predictive_model()  # builds on top of predictive model. Reuses triplet encoding

    # build denoising objective
    models = self.models
    loss1 = self._noisy_decode(models[1].encode, models[0].encode, models[1])
    loss2 = self._noisy_decode(models[1].encode, models[2].encode, models[1])
    self.loss_dn = loss2 + loss1
    self.losses = [self.loss_reco, self.loss_pred, self.loss_dist, self.loss_dn]

  def _noisy_decode(self, x1, x2, model):
    """Distort middle encoding with [<= 1/3*dist(neigbour)] and demand good reconstruction"""
    # dist = l2(x1 - x2)
    # noise = dist * self.epsilon_sphere_noise()
    # tf.stop_gradient(noise)
    noise = tf.random_normal(self.model.encode.get_shape().as_list()) * FLAGS.epsilon
    noisy_encoding = noise + self.models[1].encode
    tf.stop_gradient(noisy_encoding)  # or maybe here, who knows
    noisy_decode = interpreter.build_decoder(noisy_encoding, model.config, reuse=True, masks=model.mask_list)
    loss = 1./2 * interpreter.l2_loss(noisy_decode, self.raw_targets[1], alpha=FLAGS.beta)
    self.models += [noisy_decode]
    return loss

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

  def _init_optimizer(self):
    self.loss_total = tf.add_n(self.losses, 'loss_total')
    self.optimizer = self.optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self.train = self.optimizer.minimize(self.loss_total, global_step=self.step)


# MAIN


  def train(self):
    self.fetch_datasets()
    if FLAGS.model == AUTOENCODER:
      self.build_ae_model()
    elif FLAGS.model == PREDICTIVE:
      self.build_predictive_model()
    else:
      self.build_denoising_model()
    self._init_optimizer()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._on_training_start(sess)

      try:
        for current_epoch in range(FLAGS.max_epochs):
          start = time.time()
          full_set_blur = len(self.train_set) < 50000
          ds = self._get_blurred_dataset() if full_set_blur else self.train_set
          if FLAGS.model == AUTOENCODER:

            # Autoencoder Training
            for batch in self._batch_generator():
              summs, encoding, reconstruction, loss, _, step = sess.run(
                [self.summs_train, self.encode, self.decode, self.loss_ae, self.train_ae, self.step],
                feed_dict={self.input: batch[0], self.target: batch[1]}
              )
              self._on_batch_finish(summs, loss, batch, encoding, reconstruction)

          else:

            # Predictive and Denoising training
            for batch_indexes in self._batch_permutation_generator(len(ds)-2):
              batch = np.stack((ds[batch_indexes], ds[batch_indexes + 1], ds[batch_indexes + 2]))
              if not full_set_blur:
                batch = np.stack((
                  inp.apply_gaussian(ds[batch_indexes], sigma=self._get_blur_sigma()),
                  inp.apply_gaussian(ds[batch_indexes+1], sigma=self._get_blur_sigma()),
                  inp.apply_gaussian(ds[batch_indexes+2], sigma=self._get_blur_sigma())
                ))

              summs, loss, _ = sess.run(
                [self.summs_train, self.loss_total, self.train],
                feed_dict={self.inputs: batch, self.targets: batch})
              self._on_batch_finish(summs, loss)

          self._on_epoch_finish(current_epoch, start, sess)
        self._on_training_finish(sess)
      except KeyboardInterrupt:
        self._on_training_abort(sess)

  # @ut.timeit
  def evaluate(self, sess, take):
    digest = Bunch(encoded=None, reconstructed=None, source=None,
                   loss=.0, eval_loss=.0, dumb_loss=.0)
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    # Encode
    for i, batch in enumerate(self._batch_generator(blurred, shuffle=False)):
      encoding = self.encode.eval(feed_dict={self.input: batch[0]})
      digest.encoded = ut.concatenate(digest.encoded, encoding)
    # Save encoding for visualization
    self.embedding_assign.eval(feed_dict={self.embedding_test_ph: digest.encoded})
    self.embedding_saver.save(sess, self.get_checkpoint_path() + EMB_SUFFIX)

    # Calculate expected evaluation
    expected = digest.encoded[1:-1]*2 - digest.encoded[:-2]
    average = 0.5 * (digest.encoded[1:-1] + digest.encoded[:-2])
    digest.size = len(expected)
    # evaluation summaries
    self.summary_writer.add_summary(self.eval_summs.eval(
      feed_dict={self.blur_ph: self._get_blur_sigma()}),
      global_step=self.get_past_epochs())
    # evaluation losses
    for p in self._batch_permutation_generator(digest.size, shuffle=False):
      digest.loss      += self.eval_loss.eval(feed_dict={self.encoding: digest.encoded[p + 2], self.target: blurred[p + 2]})
      digest.eval_loss += self.eval_loss.eval(feed_dict={self.encoding: expected[p], self.target: blurred[p + 2]})
      digest.dumb_loss += self.loss_ae.eval(  feed_dict={self.input:    blurred[p], self.target: blurred[p + 2]})

    # for batch in self._batch_generator(blurred, batches=1):
    #   digest.source = batch[1][:take]
    #   digest.reconstructed = self.decode.eval(feed_dict={self.input: batch[0]})[:take]

    # Reconstruction visualizations
    for p in self._batch_permutation_generator(digest.size, shuffle=True, batches=1):
      digest.source = self.eval_decode.eval(feed_dict={self.encoding: expected[p]})[:take]
      digest.source = blurred[(p+2)[:take]]
      digest.reconstructed = self.eval_decode.eval(feed_dict={self.encoding: average[p]})[:take]
      self._eval_image_summaries(blurred[p], digest.encoded[p], average[p],  expected[p])

    digest.dumb_loss = guard_nan(digest.dumb_loss)
    digest.eval_loss = guard_nan(digest.eval_loss)
    digest.loss = guard_nan(digest.loss)
    return digest

  def _eval_image_summaries(self, blurred_batch, actual, average, expected):
    """Create Tensorboard summaries with image reconstructions"""
    noisy = expected + np.random.randn(*expected.shape) * FLAGS.epsilon

    summary = self.image_summaries['orig'].eval(feed_dict={self.input: blurred_batch})
    self.summary_writer.add_summary(summary, global_step=self.get_past_epochs())

    self._eval_image_summary('midd', average)
    # self._eval_image_summary('reco', actual)
    self._eval_image_summary('pred', expected)
    self._eval_image_summary('nois', noisy)

  def _eval_image_summary(self, name, encdoding_batch):
    summary = self.image_summaries[name].eval(feed_dict={self.encoding: encdoding_batch})
    self.summary_writer.add_summary(summary, global_step=self.get_past_epochs())

  def _add_decoding_summary(self, name, var, collection='train'):
    var = var[:FLAGS.visualiza_max]
    var = tf.concat(tf.unstack(var), axis=0)
    var = tf.expand_dims(var, dim=0)
    color_s = tf.summary.image(name, var[..., :3], max_outputs=FLAGS.visualiza_max)
    var = tf.expand_dims(var[..., 3], dim=3)
    bw_s = tf.summary.image('depth_' + name, var, max_outputs=FLAGS.visualiza_max)
    return tf.summary.merge([color_s, bw_s])


# TRAINING PROGRESS EVENTS


  def _on_training_start(self, sess):
    # Writers and savers
    self.summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    self.saver = tf.train.Saver()
    self._build_embedding_saver(sess)
    self._restore_model(sess)
    # Loss summaries
    self._build_summaries()

    self.epoch_stats = get_stats_template()
    self.stats = Bunch(
      epoch_accuracy=[],
      epoch_reconstructions=[],
      permutation=None
    )
    # if FLAGS.dev:
    #   plt.ion()
    #   plt.show()

  def _build_summaries(self):
    # losses
    with tf.name_scope('losses'):
      loss_names = ['loss_autoencoder', 'loss_predictive', 'loss_distance', 'loss_denoising']
      for i, loss in enumerate(self.losses):
        self._add_loss_summary(loss_names[i], loss)
      self._add_loss_summary('loss_total', self.loss_total)
    self.summs_train = tf.summary.merge_all('train')
    # reconstructions
    with tf.name_scope('decodings'):
      self.image_summaries = {
        'orig': self._add_decoding_summary('0_original_input', self.input),
        'reco': self._add_decoding_summary('1_reconstruction', self.eval_decode),
        'pred': self._add_decoding_summary('2_prediction', self.eval_decode),
        'midd': self._add_decoding_summary('3_averaged', self.eval_decode),
        'nois': self._add_decoding_summary('4_noisy', self.eval_decode)
      }
    # visualization
    fig = vis.get_figure()
    fig.canvas.draw()
    self.vis_placeholder = tf.placeholder(tf.uint8,  ut.fig2rgb_array(fig).shape)
    self.vis_summary = tf.summary.image('visualization', self.vis_placeholder)
    # embedding
    dists = l2(self.embedding_test[:-1] - self.embedding_test[1:])
    self.dist = dists
    embedding_d_hist = tf.summary.histogram('point distance', dists)
    embedding_trajectory = tf.summary.scalar('trajectory_length', tf.reduce_sum(dists))
    self.blur_ph = tf.placeholder(dtype=tf.float32)
    blur_summ = tf.summary.scalar('blur_sigma', self.blur_ph)
    self.eval_summs = tf.summary.merge([embedding_d_hist, embedding_trajectory, blur_summ])


  def _build_embedding_saver(self, sess):
    """To use embedding visualizer data has to be stored in variable
    since we would like to visualize TEST_SET, this variable should not affect
    common checkpoint of the model.
    Hence, we build a separate variable with a separate saver."""
    embedding_shape = [int(len(self.test_set) / FLAGS.batch_size) * FLAGS.batch_size,
                       self.encode.get_shape().as_list()[1]]
    sprite_path = os.path.join(FLAGS.logdir, 'sprite.png')
    tsv_path = os.path.join(FLAGS.logdir, 'metadata.tsv')

    self.embedding_test_ph = tf.placeholder(tf.float32, embedding_shape, name='embedding')
    self.embedding_test = tf.Variable(tf.random_normal(embedding_shape), name='test_embedding', trainable=False)
    self.embedding_assign = self.embedding_test.assign(self.embedding_test_ph)
    self.embedding_saver = tf.train.Saver(var_list=[self.embedding_test])

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = self.embedding_test.name
    embedding.sprite.image_path = sprite_path
    embedding.sprite.single_image_dim.extend([80, 80])
    embedding.metadata_path = './metadata.tsv'
    projector.visualize_embeddings(self.summary_writer, config)
    sess.run(tf.variables_initializer([self.embedding_test], name='init_embeddings'))

    # build sprite image
    ut.images_to_sprite(self.test_set, path=sprite_path)
    ut.generate_tsv(len(self.test_set), tsv_path)

  def _add_loss_summary(self, name, var, collection='train'):
    if var is not None:
      tf.summary.scalar(name, var, [collection])
      tf.summary.scalar('log_' + name, tf.log(var), [collection])

  def _restore_model(self, session):
    latest_checkpoint = tf.train.latest_checkpoint(self.get_checkpoint_path()[:-10], latest_filename='checkpoint')
    if latest_checkpoint is not None:
      latest_checkpoint = latest_checkpoint.replace(EMB_SUFFIX, '')
    ut.print_info("latest checkpoint: %s" % latest_checkpoint)
    if FLAGS.load_state and latest_checkpoint is not None:
      self.saver.restore(session, latest_checkpoint)
      ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

  def _on_batch_finish(self, summs, loss, batch=None, encoding=None, reconstruction=None):
    self.summary_writer.add_summary(summs, global_step=self.step.eval())
    self.epoch_stats.total_loss += loss

    if False:
      assert batch is not None and reconstruction is not None
      original = batch[0]
      vis.plot_reconstruction(original, reconstruction, interactive=True)

  # @ut.timeit
  def _on_epoch_finish(self, epoch, start_time, sess):
    elapsed = time.time() - start_time
    self.epoch_stats.total_loss = guard_nan(self.epoch_stats.total_loss)
    accuracy = 100000 * np.sqrt(self.epoch_stats.total_loss / np.prod(self.batch_shape) / self.epoch_size)
    # SAVE
    if is_stopping_point(epoch, FLAGS.max_epochs, FLAGS.save_every):
      self.saver.save(sess, self.get_checkpoint_path())
    # VISUALIZE
    if is_stopping_point(epoch, FLAGS.max_epochs, FLAGS.eval_every):
      evaluation = self.evaluate(sess, take=FLAGS.visualiza_max)
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
      self._save_visualization_to_summary()
    self.stats.epoch_accuracy.append(accuracy)
    self._print_epoch_info(accuracy, epoch, FLAGS.max_epochs, elapsed)
    if epoch + 1 != FLAGS.max_epochs:
      self.epoch_stats = get_stats_template()

  def _save_visualization_to_summary(self):
    image = ut.fig2rgb_array(plt.figure(num=0))
    self.summary_writer.add_summary(self.vis_summary.eval(feed_dict={self.vis_placeholder: image}))

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

  def _on_training_finish(self, sess):
    if FLAGS.max_epochs == 0:
      self._on_epoch_finish(self.get_past_epochs(), time.time(), sess)
    best_acc = np.min(self.stats.epoch_accuracy)
    ut.print_time('Best Quality: %f for %s' % (best_acc, FLAGS.net))
    self.summary_writer.close()

  def _on_training_abort(self, sess):
    print('Press ENTER to save the model')
    if getch.getch() == '\n':
      print('saving')
      self.saver.save(sess, self.get_checkpoint_path())


if __name__ == '__main__':
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  if len(args) <= 1:
    FLAGS.input_path = '../data/tmp/romb8.5.6.tar.gz'
    FLAGS.test_path = '../data/tmp/romb8.5.6.tar.gz'
    FLAGS.test_max = 2178
    FLAGS.max_epochs = 5
    FLAGS.eval_every = 1
    FLAGS.save_every = 1
    FLAGS.blur = 0.0

    # FLAGS.model = 'noise'
    # FLAGS.beta = 1.0
    # FLAGS.epsilon = .000001

  model = Autoencoder()
  if FLAGS.model == 'ae':
    FLAGS.model = AUTOENCODER
  elif 'pred' in FLAGS.model:
    print('PREDICTIVE')
    FLAGS.model = PREDICTIVE
  elif 'noi' in FLAGS.model:
    print('DENOISING')
    FLAGS.model = DENOISING
  else:
    print('Do-di-li-doo doo-di-li-don')
  model.train()
