from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json, os, re, math
import numpy as np
import utils as ut
import input as inp
import activation_functions as act
import prettytensor as pt

tf.app.flags.DEFINE_float('gradient_proportion', 5.0, 'Proportion of gradietn mixture RECO/DRAG')
tf.app.flags.DEFINE_integer('sequence_length', 25, 'size of the 1-variable-variation sequencies')

tf.app.flags.DEFINE_string('suffix', 'run', 'Suffix to use to distinguish models by purpose')
tf.app.flags.DEFINE_string('input_path', '../data/tmp_grey/romb8.2.2/img/', 'input folder')
tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_string('load_from_checkpoint', None, 'Load model state from particular checkpoint')

tf.app.flags.DEFINE_integer('save_every', 200, 'Save model state every INT epochs')
tf.app.flags.DEFINE_boolean('load_state', True, 'Load state if possible ')

tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')

tf.app.flags.DEFINE_boolean('visualize', True, 'Create visualization of decoded images along training')
tf.app.flags.DEFINE_integer('vis_substeps', 10, 'Use INT intermediate images')

tf.app.flags.DEFINE_integer('save_encodings_every', 300, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('save_visualization_every', 300, 'Save model state every INT epochs')

tf.app.flags.DEFINE_integer('blur_sigma', 0, 'Image blur maximum effect')
tf.app.flags.DEFINE_integer('blur_sigma_decrease', 1000, 'Decrease image blur every X epochs')

tf.app.flags.DEFINE_boolean('noise', True, 'apply noise to avoid discretisation')

FLAGS = tf.app.flags.FLAGS

DEV = False


def _clamp(encoding, filter):
  """Clamp part of the encoding according to filter mask
    i.e. [[0,1], [1, 2]] filter=[0, 1] => [[0.5, 1], [0.5, 2]]
    replace masked features with average over batch
    backpropagate difference between avg and actual values as an error over batch"""
  filter_neg = np.ones(len(filter), dtype=filter.dtype) - filter
  avg = encoding.mean(axis=0)*filter_neg
  grad = encoding*filter_neg - avg
  encoding = encoding * filter + avg
  return encoding, grad


def _declamp_grad(vae_grad, reco_grad, filter):
  """mask part of the gradient that corresponds to masked features
  replace masked gradient with variational objective"""
  res = vae_grad/FLAGS.gradient_proportion + reco_grad*filter
  return res


class DCIGN_model():
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
    FLAGS.batch_size = FLAGS.sequence_length
    self._weight_init = weight_init
    self._activation = activation
    self._optimizer = optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

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

    self._encode = pt.wrap(self._input)
    self._encode = self._encode.conv2d(5, 32, stride=2)
    print(self._encode.get_shape())
    self._encode = self._encode.conv2d(5, 64, stride=2)
    print(self._encode.get_shape())
    self._encode = self._encode.conv2d(5, 128, stride=2)
    print(self._encode.get_shape())
    self._encode = (self._encode.dropout(0.9).
                    flatten().
                    fully_connected(self.layer_narrow, activation_fn=None))

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

    self._decode = pt.wrap(self._clamped_variable)
    # self._decode = self._decode.reshape([FLAGS.batch_size, 1, 1, self.layer_narrow])
    print(self._decode.get_shape())
    self._decode = self._decode.fully_connected(7200)
    self._decode = self._decode.reshape([FLAGS.batch_size, 1, 1, 7200])
    self._decode = self._decode.deconv2d((10, 20), 128, edges='VALID')
    print(self._decode.get_shape())
    self._decode = self._decode.deconv2d(5, 64, stride=2)
    print(self._decode.get_shape())
    self._decode = self._decode.deconv2d(5, 32, stride=2)
    print(self._decode.get_shape())
    self._decode = self._decode.deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
    print(self._decode.get_shape())

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_scope)
    self._decoder_loss = self._decode.l2_regression(pt.wrap(self._reconstruction))
    self._opt_decoder = self._optimizer(learning_rate=FLAGS.learning_rate/FLAGS.batch_size)
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
    "generate batches corresponding to movements in particular directions"
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
    # blur
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
          encoding, = sess.run([self._encode], feed_dict={self._input: batch[0]})   # 1.1 encode: forward
          clamped_enc, vae_grad = _clamp(encoding, filter)                          # 1.2 # clamp

          sess.run(self._assign_clamped, feed_dict={self._clamped:clamped_enc})
          reconstruction, loss, clamped_gradient, _ = sess.run(               # 2.1 decode: forward+backprop
            [self._decode, self._decoder_loss, self._clamped_grad, self._train_decoder],
            feed_dict={self._clamped: clamped_enc, self._reconstruction: batch[0]})

          declamped_grad = _declamp_grad(vae_grad, clamped_gradient, filter)  # 2.2 prepare gradient
          _, step = sess.run(                                                 # 3.0 encode: backprop
            [self._train_encoder, self._step],
            feed_dict={self._input: batch[0], self._encoding: encoding-declamped_grad})          # Profit

          self._register_batch(batch, encoding, reconstruction, loss)
        self._register_epoch(current_epoch, epochs_to_train, permutation, sess)
      self._writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta

  def _register_training_start(self):
    pass

  def _register_batch(self, batch, encoding, reconstruction, loss):
    pass

  def _register_epoch(self, epoch, total_epochs, permutation, sess):
    pass


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  epochs = 300
  import sys

  model = DCIGN_model()
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


  model.train(epochs)
