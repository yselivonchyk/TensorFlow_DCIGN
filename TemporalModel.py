"""
Temporal encoder-decoder network inspired by DCIGN.
Network converts 2 subsequent frames into some reach representation.
sparse_encoding module creates low-dim representation that must be
sufficient to reconstruct img2 given representation of img1


  [img1] -> [prep] \ -----------------------[decoder] -> [img2*]
                    \                     /
                     [sparse_encoding]---/
                    /
  [img2] -> [prep] /

"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import utils as ut
import input as inp
import activation_functions as act
import Model
import time
import tensorflow.contrib.slim as slim
import visualization as vis


tf.app.flags.DEFINE_integer('stride', 1, 'Data is permuted in series of INT consecutive inputs')
FLAGS = tf.app.flags.FLAGS


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


class TemporalModel(Model.Model):
  model_id = 're11bn'

  layers = [32, 32]
  dataset = None
  permutation = None

  _batch_shape = None

  # placeholders
  _input = None
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

  def get_image_shape(self):
    return self._batch_shape[1:]

  def get_meta(self, meta=None):
    meta = super(TemporalModel, self).get_meta(meta=meta)
    return meta

  def load_meta(self, save_path):
    meta = super(TemporalModel, self).load_meta(save_path)
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.set_layer_sizes(meta['h'])
    ut.configure_folders(FLAGS, self.get_meta())
    return meta

  def build_model(self):
    """Construct encoder network: placeholders, operations, optimizer"""
    # Global step
    tf.reset_default_graph()
    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)
    # Encoder
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    img1, img2 = self._input[:,0,:], self._input[:,1,:]
    self._encode = slim.flatten(img1)
    narrow_index = np.argmin(self.layers)
    for i in range(narrow_index + 1):
      size, desc = self.layers[i], 'enc_hidden_%d' % i
      self._encode = slim.fully_connected(self._encode, size, scope=desc)
    # Decoder
    narrow, layers = self.layers[narrow_index], self.layers[narrow_index+1:]

    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    for i, size in enumerate(layers):
      start, desc = self._decode if i != 0 else self._encode,  'dec_hidden_%d' % i
      self._decode = slim.fully_connected(start, size, scope=desc, activation_fn=tf.nn.sigmoid)

    self._decode = slim.dropout(self._decode, 1.0 - FLAGS.dropout, is_training=True, scope='dropout3')
    ut.print_info('Dropout applied to the last layer of the network: %f' % (1. - FLAGS.dropout))
    self._decode = slim.fully_connected(self._decode, int(np.prod(self.get_image_shape())), scope='output')
    self._reco_loss = self._build_reco_loss(self._reconstruction)
    self._decode = tf.reshape(self._decode, self._batch_shape, name='reshape')
    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate/10)
    self._train = self._optimizer.minimize(self._reco_loss)

  def build_shift_model(self):
    """
    Emulate image reconstruction from preprocessed representation

      [img1] -> [prep] -> [decoder] -> [img1*]
    """
    # Global step
    tf.reset_default_graph()
    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)
    # Encoder
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    img1, img2 = self._input[:,0,:], self._input[:,1,:]

    net = None
    for i, l in enumerate(self.layers):
      source = img1 if i == 0 else net
      net = slim.conv2d(source, l, [3, 3], scope='conv_%d' % i, normalizer_fn=slim.batch_norm)
    net = slim.conv2d(net, 4, [1, 1], scope='conv_out', normalizer_fn=None)
    self._encode = net
    self._decode = net
    print(net)

    self._reco_loss = tf.nn.l2_loss(slim.flatten(self._decode - img1), name='reco_loss')
    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self._train = self._optimizer.minimize(self._reco_loss)

  # DATA
  @ut.timeit
  def fetch_datasets(self, activation_func_bounds):
    self.dataset = inp.read_ds_zip(FLAGS.input_path)
    if FLAGS.dev:
      self.dataset = self.dataset[:FLAGS.batch_size*5+1]
      print('Dataset cropped')

    shape = list(self.dataset.shape)
    FLAGS.epoch_size = int(shape[0] / FLAGS.batch_size)

    self._batch_shape = [FLAGS.batch_size, 2] + shape[1:]
    self.dataset = self.dataset[:int(len(self.dataset) / FLAGS.batch_size) * FLAGS.batch_size]
    self.dataset = inp.rescale_ds(self.dataset, activation_func_bounds.min, activation_func_bounds.max)

    self.test_set = inp.read_ds_zip(FLAGS.test_path)
    if FLAGS.dev:
      self.test_set = self.test_set[:FLAGS.batch_size*5+1]
      print("DEVELOPMENT MODE")
    FLAGS.test_size = int(len(self.test_set) / FLAGS.batch_size)

    test_max = int(FLAGS.test_max) if FLAGS.test_max >= 1 else int(FLAGS.test_max*len(self.test_set))
    self.test_set = self.test_set[0:test_max]
    self.test_set = inp.rescale_ds(self.test_set, self._activation.min, self._activation.max)

  def _batch_generator(self, dataset=None, shuffle=True):
    """Returns BATCH_SIZE of couples of subsequent images"""
    dataset = dataset if dataset is not None else self._get_blurred_dataset()
    self.permutation = np.arange(len(dataset) - 1)
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    total_batches = int(len(self.permutation) / FLAGS.batch_size)

    for i in range(total_batches):
      batch_indexes = self.permutation[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
      batch = np.stack((dataset[batch_indexes], dataset[batch_indexes+1]), axis=1)
      yield batch, batch

  def set_layer_sizes(self, h):
    if isinstance(h, str):
      ut.print_info('new layer sizes: %s' % h)
      h = h.replace('/', '|')
      h = list(map(int, h.split('|')))
    self.layers = h

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    ut.configure_folders(FLAGS, meta)

    self.fetch_datasets(self._activation)
    self.build_shift_model()
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

          self._register_batch(loss, batch, encoding, reconstruction, step)
        self._register_epoch(current_epoch, epochs_to_train, time.time()-start, sess)
      self._writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta, self._stats['epoch_accuracy']

  def evaluate(self, sess, take):
    encoded, reconstructed = None, None
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    for batch in self._batch_generator(blurred, shuffle=False):
      encoding, reconstruction = sess.run(
        [self._encode, self._decode],
        feed_dict={self._input: batch[0]})
      encoded = ut.concatenate(encoded, encoding)
      vis.plot_reconstruction(batch[0][:,0], reconstruction, interactive=True)
      # random images for reconstruction
      if FLAGS.test_size > take:
        reconstructed = ut.concatenate(reconstructed, np.asarray([reconstruction[0]]))
      else:
        reconstructed = ut.concatenate(reconstructed, reconstruction, take=take)
    if FLAGS.test_size > take:
      reconstructed = np.random.permutation(reconstructed)[:take]
    return encoded, reconstructed, blurred

  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    if Model.is_stopping_point(epoch, total_epochs, FLAGS.save_every):
      self._saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self._epoch_stats['total_loss'] / np.prod(self._batch_shape) / FLAGS.epoch_size)

    if Model.is_stopping_point(epoch, total_epochs, FLAGS.save_encodings_every):
      digest = self.evaluate(sess, take=self.MAX_IMAGES)
      data = {
        'rec': np.asarray(digest[1]),
        'blu': np.asarray(digest[2][:self.MAX_IMAGES])
      }
      meta = {'suf': 'encodings', 'e': '%06d' % int(self.get_past_epochs()), 'er': int(accuracy)}
      vis.plot_reconstruction(data['blu'], data['rec'], meta)

    self._stats['epoch_accuracy'].append(accuracy)
    self.print_epoch_info(accuracy, epoch, total_epochs, elapsed)
    if epoch + 1 != total_epochs:
      self._epoch_stats = self._get_stats_template()


if __name__ == '__main__':
  import sys

  model = TemporalModel()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  print(args)
  FLAGS.blur_decrease = 1000
  FLAGS.blur = 1.0
  if len(args) <= 1:
    FLAGS.max_epochs = 50
    FLAGS.save_encodings_every = 1
    FLAGS.dev = True
    ut.print_info('DEV mode', color=33)
    FLAGS.blur = 0.0

  if 'h' in args:
    model.set_layer_sizes(args['h'])
  model.train(FLAGS.max_epochs)