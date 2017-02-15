"""
                     (2*h2-h3)= (p1)  -> [dec] -> img1*
  [img1] -> [enc] -> (h1)             -> [dec] -> img1*
  [img2] -> [enc] -> (h2)             -> [dec] -> img2*
  [img3] -> [enc] -> (h3)             -> [dec] -> img3*
                     (2*h2-h1)= (p3)  -> [dec] -> img3*

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
import getch

tf.app.flags.DEFINE_integer('stride', 1, 'Data is permuted in series of INT consecutive inputs')
tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
FLAGS = tf.app.flags.FLAGS


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


class TemporalPredictiveModel(Model.Model):
  model_id = 'predictiv'

  layers = [40, 6, 40]
  layer_narrow = 1

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
    return self._batch_shape[2:]

  def get_decoding_shape(self):
    return self._batch_shape[:1] + self._batch_shape[2:]

  def get_meta(self, meta=None):
    meta = super(TemporalPredictiveModel, self).get_meta(meta=meta)
    meta['al'] = FLAGS.alpha
    return meta

  def load_meta(self, save_path):
    meta = super(TemporalPredictiveModel, self).load_meta(save_path)
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.set_layer_sizes(meta['h'])
    ut.configure_folders(FLAGS, self.get_meta())
    return meta

  def build_predictive_model(self):
    """
    Emulate image reconstruction from preprocessed representation

      [img1] -> [prep] -> [decoder] -> [img1*]
    """
    # Global step
    tf.reset_default_graph()
    # self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)
    # Encoder
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    img1, img2, img3 = self._input[:, 0, :], self._input[:, 1, :], self._input[:, 2, :]
    encodings = list(map(self._encoder, [img1, img2, img3]))
    encodings += [2 * encodings[1] - encodings[2], 2 * encodings[1] - encodings[0]]
    decodings = list(map(self._decoder, encodings))

    l1 = tf.nn.l2_loss(decodings[0] - slim.flatten(img1), name='reco1_loss')
    l2 = tf.nn.l2_loss(decodings[1] - slim.flatten(img2), name='reco2_loss')
    l3 = tf.nn.l2_loss(decodings[2] - slim.flatten(img3), name='reco3_loss')
    p1 = tf.nn.l2_loss(decodings[3] - slim.flatten(img1), name='pred1_loss')
    p3 = tf.nn.l2_loss(decodings[4] - slim.flatten(img3), name='pred3_loss')
    loss = l1 + l2 + l3 + FLAGS.alpha * (p1 + p3)

    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self._train = self._optimizer.minimize(loss)

    self._reco_loss = l2
    self._decode = tf.reshape(decodings[4], self.get_decoding_shape(), name='reshape')
    self._encode = encodings[1]

  _encoder_initialized = False

  def _encoder(self, img):
    encoding = slim.flatten(img)
    for i in range(self.layer_narrow + 1):
      size, desc = self.layers[i], 'enc_%d' % i
      encoding = slim.fully_connected(encoding, size, activation_fn=tf.nn.sigmoid, scope=desc,
                                      reuse=self._encoder_initialized)
    self._encoder_initialized = True
    return encoding

  _decoder_initialized = False

  def _decoder(self, enc):
    decoding = None
    for i, size in enumerate(self.layers[self.layer_narrow + 1:]):
      decoding = enc if decoding is None else decoding
      desc = 'dec_%d' % i
      decoding = slim.fully_connected(decoding, size, scope=desc, activation_fn=tf.nn.sigmoid,
                                      reuse=self._decoder_initialized)
    decoding = slim.fully_connected(decoding, int(np.prod(self.get_image_shape())), scope='output',
                                    reuse=self._decoder_initialized)
    self._decoder_initialized = True
    return decoding

  # DATA
  @ut.timeit
  def fetch_datasets(self, activation_func_bounds):
    self.dataset = inp.read_ds_zip(FLAGS.input_path)
    if FLAGS.dev:
      self.dataset = self.dataset[:FLAGS.batch_size * 5 + 1]
      print('Dataset cropped')

    shape = list(self.dataset.shape)
    FLAGS.epoch_size = int(shape[0] / FLAGS.batch_size)

    self._batch_shape = [FLAGS.batch_size, 3] + shape[1:]
    self.dataset = self.dataset[:int(len(self.dataset) / FLAGS.batch_size) * FLAGS.batch_size]
    self.dataset = inp.rescale_ds(self.dataset, activation_func_bounds.min, activation_func_bounds.max)

    self.test_set = inp.read_ds_zip(FLAGS.test_path)
    if FLAGS.dev:
      self.test_set = self.test_set[:FLAGS.batch_size * 5 + 1]
      print("DEVELOPMENT MODE")
    FLAGS.test_size = int(len(self.test_set) / FLAGS.batch_size)

    test_max = int(FLAGS.test_max) if FLAGS.test_max >= 1 else int(FLAGS.test_max * len(self.test_set))
    self.test_set = self.test_set[0:test_max]
    self.test_set = inp.rescale_ds(self.test_set, self._activation.min, self._activation.max)

  def _batch_generator(self, dataset=None, shuffle=True):
    """Returns BATCH_SIZE of couples of subsequent images"""
    dataset = dataset if dataset is not None else self._get_blurred_dataset()
    self.permutation = np.arange(len(dataset) - 2)
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    total_batches = int(len(self.permutation) / FLAGS.batch_size)

    for i in range(total_batches):
      batch_indexes = self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
      batch = np.stack((dataset[batch_indexes], dataset[batch_indexes + 1], dataset[batch_indexes + 2]), axis=1)
      yield batch, batch

  def set_layer_sizes(self, h):
    if isinstance(h, str):
      ut.print_info('new layer sizes: %s' % h)
      h = h.replace('/', '|')
      h = list(map(int, h.split('|')))
    self.layers = h
    self.layer_narrow = np.argmin(h)

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    ut.configure_folders(FLAGS, meta)

    self.fetch_datasets(self._activation)
    self.build_predictive_model()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._register_training_start(sess)
      self.restore_model(sess)

      # MAIN LOOP
      try:
        for current_epoch in xrange(epochs_to_train):
          start = time.time()
          for batch in self._batch_generator():
            encoding, reconstruction, loss, _, step = sess.run(
              [self._encode, self._decode, self._reco_loss, self._train, self._step],
              feed_dict={self._input: batch[0]})
            self._register_batch(loss, batch, encoding, reconstruction, step)
            # ut.print_model_info(trainable=True)
          self._register_epoch(current_epoch, epochs_to_train, time.time() - start, sess)
        self._writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        meta = self._register_training()
      except KeyboardInterrupt:
        self._try_save(sess)

    return meta, self._stats['epoch_accuracy']

  def evaluate(self, sess, take):
    encoded, reconstructed, data = None, None, None
    blurred = inp.apply_gaussian(self.test_set, self._get_blur_sigma())
    for batch in self._batch_generator(blurred, shuffle=False):
      encoding = sess.run([self._encode], feed_dict={self._input: batch[0]})[0]
      encoded = np.concatenate((encoded, encoding)) if encoded is not None else encoding

    for batch in self._batch_generator(blurred):
      reco = sess.run([self._decode], feed_dict={self._input: batch[0]})[0]
      break
    return encoded, reco[:take], batch[0][:,1,:][:take]

  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    if Model.is_stopping_point(epoch, total_epochs, FLAGS.save_every):
      self._saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self._epoch_stats['total_loss'] / np.prod(self._batch_shape) / FLAGS.epoch_size)

    if Model.is_stopping_point(epoch, total_epochs, FLAGS.save_encodings_every):
      encoding, reconstruction, source = self.evaluate(sess, take=self.MAX_IMAGES)
      print(encoding.shape, reconstruction.shape, source.shape)
      data = {
        'enc': np.asarray(encoding),
        'rec': np.asarray(reconstruction),
        'blu': np.asarray(source)
      }
      meta = {'suf': 'encodings', 'e': '%06d' % int(self.get_past_epochs()), 'er': int(accuracy)}
      projection_file = ut.to_file_name(meta, FLAGS.save_path)
      np.save(projection_file, data)
      vis.plot_encoding_crosssection(encoding, FLAGS.save_path, meta, source, reconstruction, interactive=FLAGS.dev)

    self._stats['epoch_accuracy'].append(accuracy)
    self.print_epoch_info(accuracy, epoch, total_epochs, elapsed)
    if epoch + 1 != total_epochs:
      self._epoch_stats = self._get_stats_template()

  def _try_save(self, sess):
    print('Press ENTER to save model')
    if getch.getch() == '\n':
      print('saving')
      self._saver.save(sess, self.get_checkpoint_path())


if __name__ == '__main__':
  import sys

  model = TemporalPredictiveModel()
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
