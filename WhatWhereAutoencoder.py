from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import utils as ut
import network_utils as nut
import activation_functions as act
import Model
import time
import matplotlib.pyplot as plt
import io
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 2, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')
FLAGS = tf.app.flags.FLAGS

CPU_ONLY = True


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


def _draw_subplot(input, index, name):
  subplot = plt.subplot(index)
  subplot.set_title(name)
  subplot.axis('off')
  subplot.imshow(np.squeeze(input))


class WhatWhereAutoencoder(Model.Model):
  model_id = 'pred_conv2'

  _batch_shape = None

  # placeholders
  _input = None
  _reconstruction = None

  # operations
  _encode = None
  _decode = None
  # train
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

  def get_image_shape(self):
    return self._batch_shape[2:]

  def build_mnist_model(self, naive=False):
    tf.reset_default_graph()

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)

    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')

    # Encoder. (16)5c-(32)3c-Xp
    net = slim.conv2d(self._input, 16, [5, 5])
    net = slim.conv2d(net, 32, [3, 3])
    if not naive and not CPU_ONLY:
      self._encode, mask = nut.max_pool_with_argmax(net, FLAGS.pool_size)
      net = nut.unpool(self._encode, mask, stride=FLAGS.pool_size)
    else:
      self._encode = slim.max_pool2d(net, kernel_size=[FLAGS.pool_size, FLAGS.pool_size])
      net = nut.upsample(self._encode, stride=FLAGS.pool_size)

    net = slim.conv2d_transpose(net, 16, [3, 3])
    net = slim.conv2d_transpose(net, 1, [5, 5])
    self._decode = net

    l2rec = tf.nn.l2_loss(slim.flatten(self._input) - slim.flatten(net))
    self._reco_loss = l2rec

    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self._train = self._optimizer.minimize(l2rec)
    return self._train, self._encode, self._decode

  @ut.timeit
  def fetch_datasets(self):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    self.dataset = mnist.train.images.reshape((55000, 28, 28, 1))
    self._batch_shape = [FLAGS.batch_size, 28, 28, 1]
    print(self.dataset.shape)

  def _batch_generator(self, dataset=None, shuffle=True):
    """Returns BATCH_SIZE of images"""
    dataset = dataset if dataset is not None else self._get_blurred_dataset()
    self.permutation = np.arange(len(dataset) - 2)
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    total_batches = int(len(self.permutation) / FLAGS.batch_size)

    for i in range(total_batches):
      batch_indexes = self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
      yield dataset[batch_indexes]

  # TRAIN

  def train(self, epochs_to_train=5):
    self.fetch_datasets()
    self.build_mnist_model()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._register_training_start(sess)
      self.restore_model(sess)

      ut.print_model_info(trainable=True)

      # MAIN LOOP
      for current_epoch in xrange(epochs_to_train):
        start = time.time()
        for batch in self._batch_generator():
          encoding, reconstruction, loss, _, step = sess.run(
            [self._encode, self._decode, self._reco_loss, self._train, self._step],
            feed_dict={self._input: batch})
          self._register_batch(loss, batch, encoding, reconstruction, step)
        self._register_epoch(current_epoch, epochs_to_train, time.time() - start, sess)
      # meta = self._register_training()

  def _register_training_start(self, sess):
    self.summary_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    self._epoch_stats = self._get_stats_template()

  def _build_reconstruction_comparison(self, input, reco, reference_reco=None, interactive=False):
    plt.figure()
    print(input.shape)
    _draw_subplot(input, 131, 'input')
    _draw_subplot(reco, 132, 'reconstruction')
    reference_reco = reference_reco if reference_reco is not None else reco
    _draw_subplot(reference_reco, 133, 'naive')

    buf = io.BytesIO()
    if interactive:
      plt.show()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

  def _register_batch(self, loss, batch=None, encoding=None, reconstruction=None, step=None):
    self._epoch_stats['total_loss'] += loss
    if step % 10 == 0:
      key = 'period_start' if 'period_start' in self._epoch_stats else 'start'
      passed = time.time() - self._epoch_stats[key]
      self._epoch_stats['period_start'] = time.time()
      print('\r step: %d/%d, batch_t: %.3f' % (step, 55000/FLAGS.batch_size, passed/100), end='')

      # Convert PNG buffer to TF image
      png_buffer = self._build_reconstruction_comparison(batch[0], reconstruction[0])
      image = tf.image.decode_png(png_buffer.getvalue(), channels=4)
      image = tf.expand_dims(image, 0)
      summary_op = tf.summary.image("what_where", image)
      self.summary_writer.add_summary(summary_op.eval())

  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    print(self._epoch_stats['total_loss'])
    if epoch + 1 != total_epochs:
      self._epoch_stats = self._get_stats_template()


if __name__ == '__main__':
  FLAGS.blur = 0.0
  model = WhatWhereAutoencoder()
  model.train(FLAGS.max_epochs)
