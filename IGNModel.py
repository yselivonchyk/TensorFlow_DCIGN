"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json, os, re, math
import numpy as np
import utils as ut
import input as inp
import tools.checkpoint_utils as ch_utils
import activation_functions as act
import visualization as vis
import prettytensor as pt
import Model as m

tf.app.flags.DEFINE_float('gradient_proportion', 5.0, 'Proportion of gradietn mixture RECO/DRAG')
tf.app.flags.DEFINE_integer('sequence_length', 25, 'size of the 1-variable-variation sequencies')

FLAGS = tf.app.flags.FLAGS

DEV = False


def _clamp(encoding, filter):
  filter_neg = np.ones(len(filter), dtype=filter.dtype) - filter
  # print('\nf_n', filter_neg)
  avg = encoding.mean(axis=0)*filter_neg
  # print('avg', avg, encoding[0])
  # print('avg', encoding.mean(axis=0), encoding[-1], encoding[-1]-avg)
  grad = encoding*filter_neg - avg
  encoding = encoding * filter + avg
  # print('enc', encoding[0], encoding[1])
  # print(np.hstack((encoding, grad)))
  # print('vae', grad[0], grad[1])
  return encoding, grad


def _declamp_grad(vae_grad, reco_grad, filter):
  # print('vae, reco', np.abs(vae_grad).mean(), np.abs((reco_grad*filter)).mean())
  res = vae_grad/FLAGS.gradient_proportion + reco_grad*filter
  # res = vae_grad + reco_grad*filter
  #print('\nvae: %s\nrec: %s\nres %s' % (ut.print_float_list(vae_grad[1]),
  #                                      ut.print_float_list(reco_grad[1]),
  #                                      ut.print_float_list(res[0])))
  return res


class IGNModel(m.Model):
  model_id = 'ign'
  decoder_scope = 'dec'
  encoder_scope = 'enc'

  layer_narrow = 2
  layer_encoder = 40
  layer_decoder = 40

  _image_shape = None
  _batch_shape = None

  # placeholders
  _input = None
  _encoding = None

  _clamped = None
  _reconstruction = None
  _clamped_grad = None

  # variables
  _clamped_variable = None

  # operations
  _encode = None
  _encoder_loss = None
  _opt_encoder = None
  _train_encoder = None

  _decode = None
  _decoder_loss = None
  _opt_decoder = None
  _train_decoder = None

  _step = None
  _current_step = None
  _visualize_op = None

  def __init__(self,
               weight_init=None,
               activation=act.sigmoid,
               optimizer=tf.train.AdamOptimizer):
    super(IGNModel, self).__init__()
    FLAGS.batch_size = FLAGS.sequence_length
    self._weight_init = weight_init
    self._activation = activation
    self._optimizer = optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  def get_meta(self, meta=None):
    meta = super(IGNModel, self).get_meta(meta=meta)
    meta['div'] = FLAGS.gradient_proportion
    return meta

  def load_meta(self, save_path):
    meta = super(IGNModel, self).load_meta(save_path)
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.layer_encoder = meta['h'][0]
    self.layer_narrow = meta['h'][1]
    self.layer_decoder = meta['h'][2]
    FLAGS.gradient_proportion = float(meta['div'])
    ut.configure_folders(FLAGS, self.get_meta())
    return meta

  # MODEL

  def build_model(self):
    tf.reset_default_graph()
    self._batch_shape = inp.get_batch_shape(FLAGS.batch_size, FLAGS.input_path)
    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)
    with pt.defaults_scope(activation_fn=self._activation.func):
      with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope(self.encoder_scope):
          self._build_encoder()
        with tf.variable_scope(self.decoder_scope):
          self._build_decoder()

  def _build_encoder(self):
    """Construct encoder network: placeholders, operations, optimizer"""
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    self._encoding = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow), name='encoding')

    self._encode = (pt.wrap(self._input)
                    .flatten()
                    .fully_connected(self.layer_encoder, name='enc_hidden')
                    .fully_connected(self.layer_narrow, name='narrow'))

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope)
    self._encoder_loss = self._encode.l1_regression(pt.wrap(self._encoding))
    ut.print_info('new learning rate: %.8f (%f)' % (FLAGS.learning_rate/FLAGS.batch_size, FLAGS.learning_rate))
    self._opt_encoder = self._optimizer(learning_rate=FLAGS.learning_rate/FLAGS.batch_size)
    self._train_encoder = self._opt_encoder.minimize(self._encoder_loss)

  def _build_decoder(self, weight_init=tf.truncated_normal):
    """Construct decoder network: placeholders, operations, optimizer,
    extract gradient back-prop for encoding layer"""
    self._clamped = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow))
    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    clamped_init = np.zeros((FLAGS.batch_size, self.layer_narrow), dtype=np.float32)
    self._clamped_variable = tf.Variable(clamped_init, name='clamped')
    self._assign_clamped = tf.assign(self._clamped_variable, self._clamped)

    # http://stackoverflow.com/questions/40194389/how-to-propagate-gradient-into-a-variable-after-assign-operation
    self._decode = (
      pt.wrap(self._clamped_variable)
        .fully_connected(self.layer_decoder, name='decoder_1')
        .fully_connected(np.prod(self._image_shape), init=weight_init, name='output')
        .reshape(self._batch_shape))

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_scope)
    self._decoder_loss = self._build_reco_loss(self._reconstruction)
    self._opt_decoder = self._optimizer(learning_rate=FLAGS.learning_rate)
    self._train_decoder = self._opt_decoder.minimize(self._decoder_loss)

    self._clamped_grad, = tf.gradients(self._decoder_loss, [self._clamped_variable])

  # DATA

  def fetch_datasets(self, activation_func_bounds):
    original_data, filters = inp.get_images(FLAGS.input_path)
    assert len(filters) == len(original_data)
    original_data, filters = self.bloody_hack_filterbatches(original_data, filters)
    ut.print_info('shapes. data, filters: %s' % str((original_data.shape, filters.shape)))

    original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)
    self._image_shape = inp.get_image_shape(FLAGS.input_path)

    if DEV:
      original_data = original_data[:300]

    self.epoch_size = math.ceil(len(original_data) / FLAGS.batch_size)
    self.test_size = math.ceil(len(original_data) / FLAGS.batch_size)
    return original_data, filters

  def bloody_hack_filterbatches(self, original_data, filters):
    print(filters)
    survivers = np.zeros(len(filters), dtype=np.uint8)
    j, prev = 0, None
    for _, f in enumerate(filters):
      if prev is None or prev[0] == f[0] and prev[1] == f[1]:
        j += 1
      else:
        k = j // FLAGS.batch_size
        for i in range(k):
          start = _ - j + math.ceil(j / k * i)
          survivers[start:start + FLAGS.batch_size] += 1
        # print(j, survivers[_-j:_])
        j = 0
      prev = f
    original_data = np.asarray([x for i, x in enumerate(original_data) if survivers[i] > 0])
    filters = np.asarray([x for i, x in enumerate(filters) if survivers[i] > 0])
    return original_data, filters

  def _get_epoch_dataset(self):
    ds, filters = self._get_blurred_dataset(), self._filters
    # permute
    (train_set, filters), permutation = inp.permute_data_in_series((ds, filters), FLAGS.batch_size, allow_shift=False)
    # construct feed
    feed = pt.train.feed_numpy(FLAGS.batch_size, train_set, filters)
    return feed, permutation

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    # return meta, np.random.randn(epochs_to_train)
    ut.configure_folders(FLAGS, meta)

    self._dataset, self._filters = self.fetch_datasets(self._activation)
    self.build_model()
    self._register_training_start()

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      self._saver = tf.train.Saver()

      if FLAGS.load_state and os.path.exists(self.get_checkpoint_path()):
        self._saver.restore(sess, self.get_checkpoint_path())
        ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

      # MAIN LOOP
      for current_epoch in xrange(epochs_to_train):

        feed, permutation = self._get_epoch_dataset()
        for _, batch in enumerate(feed):
          filter = batch[1][0]
          assert batch[1][0,0] == batch[1][-1,0]
          encoding, = sess.run([self._encode], feed_dict={self._input: batch[0]})   # 1.1 encode forward
          clamped_enc, vae_grad = _clamp(encoding, filter)                          # 1.2 # clamp

          sess.run(self._assign_clamped, feed_dict={self._clamped:clamped_enc})
          reconstruction, loss, clamped_gradient, _ = sess.run(          # 2.1 decode forward+backward
            [self._decode, self._decoder_loss, self._clamped_grad, self._train_decoder],
            feed_dict={self._clamped: clamped_enc, self._reconstruction: batch[0]})

          declamped_grad = _declamp_grad(vae_grad, clamped_gradient, filter) # 2.2 prepare gradient
          _, step = sess.run(                                            # 3.0 encode backward path
            [self._train_encoder, self._step],
            feed_dict={self._input: batch[0], self._encoding: encoding-declamped_grad})          # Profit

          self._register_batch(batch, encoding, reconstruction, loss)
        self._register_epoch(current_epoch, epochs_to_train, permutation, sess)
      self._writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta, self._stats['epoch_accuracy']



if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  epochs = 300
  import sys

  model = IGNModel()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  if len(args) == 0:
    global DEV
    DEV = False
    print('DEVELOPMENT MODE ON')
  print(args)
  if 'epochs' in args:
    epochs = int(args['epochs'])
    ut.print_info('epochs: %d' % epochs, color=36)
  if 'sigma' in args:
    FLAGS.sigma = int(args['sigma'])
  if 'suffix' in args:
    FLAGS.suffix = args['suffix']
  if 'input' in args:
    parts = FLAGS.input_path.split('/')
    parts[-3] = args['input']
    FLAGS.input_path = '/'.join(parts)
    ut.print_info('input %s' % FLAGS.input_path, color=36)
  if 'h' in args:
    layers = list(map(int, args['h'].split('/')))
    ut.print_info('layers %s' % str(layers), color=36)
    model.set_layer_sizes(layers)
  if 'divider' in args:
    FLAGS.drag_divider = float(args['divider'])
  if 'lr' in args:
    FLAGS.learning_rate = float(args['lr'])


  model.train(epochs)
