"""
This class contains the code to recreate FullyConnected model with TF12\slim
using checkpoint of TF8\prettytensor.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import numpy as np
import utils as ut
import input as inp
import activation_functions as act
import Model
import time
import tools.checkpoint_utils as chut
import tensorflow.contrib.slim as slim


tf.app.flags.DEFINE_integer('stride', 1, 'Data is permuted in series of INT consecutive inputs')
FLAGS = tf.app.flags.FLAGS

DEV = False


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


class TemporalModel(Model.Model):
  model_id = 'fc'

  layers = [3000,50,5,50]
  layer_narrow = 2

  _batch_shape = None

  # placeholders
  _input = None
  _encoding = None
  _reconstruction = None

  # operations
  _encode = None
  _decode = None
  _reco_loss = None
  _optimizer = None
  _train = None

  _step = None
  _current_step = None
  _visualize_op = None

  def __init__(self,
               weight_init=None,
               activation=act.sigmoid,
               optimizer=tf.train.AdamOptimizer):
    self._weight_init = weight_init
    self._activation = activation
    self._optimizer_constructor = optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def get_layer_info(self):
    return self.layers

  def get_meta(self, meta=None):
    meta = super(TemporalModel, self).get_meta(meta=meta)
    # meta['seq'] = FLAGS.stride
    return meta

  def load_meta(self, save_path):
    meta = super(TemporalModel, self).load_meta(save_path)
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.set_layer_sizes(meta['h'])
    # FLAGS.stride = int(meta['str']) if 'str' in meta else 2
    ut.configure_folders(FLAGS, self.get_meta())
    return meta

  @ut.timeit
  def build_model(self):
    """Construct encoder network: placeholders, operations, optimizer"""
    # Global step
    tf.reset_default_graph()
    self._current_step = tf.Variable(self.get_initializers('global_step'), trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)
    # Encoder
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    self._encode = slim.flatten(self._input)
    for i in range(self.layer_narrow + 1):
      size, desc = self.layers[i], 'enc_hidden_%d' % i
      w_init, b_init = self.get_initializers('enc', i)
      # self._encode = slim.fully_connected(self._encode, size, scope=desc)
      self._encode = slim.fully_connected(self._encode, num_outputs=size, scope=desc,
                                          activation_fn=tf.nn.sigmoid,
                                          weights_initializer=w_init,
                                          biases_initializer=b_init)
    # Decoder
    narrow, layers = self.layers[self.layer_narrow], self.layers[self.layer_narrow+1:]

    self._encoding = tf.placeholder(tf.float32, (FLAGS.batch_size, narrow), name='encoding')
    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    for i, size in enumerate(layers):
      start = self._decode if i != 0 else self._encode
      w_init, b_init = self.get_initializers('dec', i)
      desc = 'dec_hidden_%d' % i
      print(size)
      self._decode = slim.fully_connected(start, size, scope=desc,
                                          activation_fn=tf.nn.sigmoid,
                                          weights_initializer=w_init,
                                          biases_initializer=b_init)

    self._decode = slim.dropout(self._decode, 1.0 - FLAGS.dropout, is_training=True, scope='dropout3')
    ut.print_info('Dropout applied to the last layer of the network: %f' % (1. - FLAGS.dropout))
    w_init, b_init = self.get_initializers('output', i)
    # self._decode = slim.fully_connected(self._decode, int(np.prod(self._image_shape)), scope='output')
    self._decode = slim.fully_connected(self._decode, int(np.prod(self._image_shape)), scope='output',
                                        activation_fn=tf.nn.sigmoid,
                                        weights_initializer=w_init,
                                        biases_initializer=b_init)
    self._reco_loss = self._build_reco_loss(self._reconstruction)
    self._decode = tf.reshape(self._decode, self._batch_shape, name='reshape')
    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate/10000)
    self._train = self._optimizer.minimize(self._reco_loss)

  def get_initializers(self, name, index=None):
    self.old_checkpoint = self.get_checkpoint_path()
    if name == 'global_step':
      print('global_step', chut.load_variable(self.old_checkpoint, name))
      return chut.load_variable(self.old_checkpoint, name)
    name = '%s/enc_hidden_%d/' % (name, index) if name != 'output' else 'dec/output/'
    w = chut.load_variable(self.old_checkpoint, name + 'weights')
    b = np.reshape(chut.load_variable(self.old_checkpoint, name + 'bias'), (len(w[0])))
    print(name + 'weights', w.shape, b.shape, index)
    return tf.constant_initializer(w), tf.constant_initializer(b)

  # DATA
  @ut.timeit
  def fetch_datasets(self, activation_func_bounds):
    self.dataset = inp.read_ds_zip(FLAGS.input_path)
    if DEV:
      self.dataset = self.dataset[:FLAGS.batch_size*5]
      print('Dataset cropped')

    shape = list(self.dataset.shape)
    FLAGS.epoch_size = int(shape[0] / FLAGS.batch_size)

    self._batch_shape = shape
    self._batch_shape[0] = FLAGS.batch_size
    self.dataset = self.dataset[:int(len(self.dataset) / FLAGS.batch_size) * FLAGS.batch_size]
    self.dataset = inp.rescale_ds(self.dataset, activation_func_bounds.min, activation_func_bounds.max)
    self._image_shape = list(self.dataset.shape)[1:]

    self.test_set = inp.read_ds_zip(FLAGS.test_path)
    if DEV:
      self.test_set = self.test_set[:FLAGS.batch_size*5]
      print()
    test_max = int(FLAGS.test_max) if FLAGS.test_max >= 1 else int(FLAGS.test_max*len(self.test_set))
    self.test_set = self.test_set[0:test_max]
    self.test_set = inp.rescale_ds(self.test_set, self._activation.min, self._activation.max)

  def _batch_generator(self, dataset=None, shuffle=True):
    dataset = dataset if dataset is not None else self._get_blurred_dataset()
    permutation = np.arange(len(dataset))
    permutation = permutation if not shuffle else np.random.permutation(permutation)
    total_batches = int(len(dataset) / FLAGS.batch_size)
    for i in range(total_batches):
      batch_indexes = permutation[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
      batch = dataset[batch_indexes]
      yield batch, batch

  def set_layer_sizes(self, h):
    if isinstance(h, str):
      ut.print_info('new layer sizes: %s' % h)
      h = h.replace('/', '|')
      h = list(map(int, h.split('|')))
    self.layers = h
    self.layer_narrow = np.argmin(h)
    print(self.layers, self.layer_narrow)

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    ut.configure_folders(FLAGS, meta)

    self.fetch_datasets(self._activation)
    self.build_model()
    self._register_training_start()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_model(sess)

      # MAIN LOOP
      for current_epoch in xrange(epochs_to_train):
        start = time.time()
        for batch in self._batch_generator():
          encoding, reconstruction, loss, _, step = sess.run(
            [self._encode, self._decode, self._reco_loss, self._train, self._step],
            feed_dict={self._input: batch[0], self._reconstruction: batch[0]})

          self._register_batch(loss)
        self._register_epoch(current_epoch, epochs_to_train, time.time()-start, sess)
      self._writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta, self._stats['epoch_accuracy']

  def evaluate(self, sess, take):
    encoded, reconstructed = None, None
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    for batch in self._batch_generator(blurred, shuffle=None):
      encoding, reconstruction = sess.run(
        [self._encode, self._decode],
        feed_dict={self._input: batch[0]})
      encoded = self._concatenate(encoded, encoding)
      reconstructed = self._concatenate(reconstructed, reconstruction, take=take)
    return encoded, reconstructed, blurred

  @staticmethod
  def _concatenate(x, y, take=None):
    if take is not None and x is not None and len(x) >= take:
      return x
    if x is None:
      res = y
    else:
      res = np.concatenate((x, y))
    return res[:take] if take is not None else res


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  # print('\n'.join([str(x) for x in chut.list_variables('./tmp/run_do__a|s__do|0.__h|3000|50|5|50__inp|grid03.14/-9999.chpt') if 'Adam' not in x[0]]))
  import sys
  print(tf.__version__)

  model = TemporalModel()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  print(args)
  if len(args) <= 1:
    DEV = True
    ut.print_info('DEV mode', color=33)
    FLAGS.blur = 0.0

  if 'suffix' in args:
    FLAGS.suffix = args['suffix']
  # if 'input' in args:
  #   if args['input'][0] != '/':
  #     args['input'] = '/' + args['input']
  #   if 'tmp' not in args['input']:
  #     args['input'] = '/tmp' + args['input']
  #   args['input'] = '../data' + args['input']
  #   FLAGS.input_path = args['input']
  #   ut.print_info('input: %s' % FLAGS.input_path, color=36)
  if 'h' in args:
    model.set_layer_sizes(args['h'])

  all_data = [x[0] for x in os.walk( '../data/tmp_grey/') if 'img' in x[0]]
  # for _, path in enumerate(all_data):
  #   print(path)
  #   FLAGS.input_path = path
  model.train(FLAGS.max_epochs)