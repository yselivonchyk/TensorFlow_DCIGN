from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import utils as ut
import activation_functions as act
import Model
import time
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 2, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')
FLAGS = tf.app.flags.FLAGS


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


# Thank you, @https://github.com/Pepslee
def unpool(updates, mask, ksize=[1, 2, 2, 1]):
  input_shape = updates.get_shape().as_list()
  #  calculation new shape
  output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
  # calculation indices for batch, height, width and feature maps
  one_like_mask = tf.ones_like(mask)
  batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
  b = one_like_mask * batch_range
  y = mask // (output_shape[2] * output_shape[3])
  x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
  feature_range = tf.range(output_shape[3], dtype=tf.int64)
  f = one_like_mask * feature_range
  # transpose indices & reshape update values to one dimension
  updates_size = tf.size(updates)
  indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
  values = tf.reshape(updates, [updates_size])
  ret = tf.scatter_nd(indices, values, output_shape)
  return ret


class TemporalPredictiveModel(Model.Model):
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

  def build_mnist_model(self):
    tf.reset_default_graph()

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)

    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')

    # Encoder. (16)5c-(32)3c-Xp
    net = slim.conv2d(self._input, 16, [5, 5])
    net = slim.conv2d(net, 32, [3, 3])
    net = slim.max_pool2d(net, kernel_size=[FLAGS.pool_size, FLAGS.pool_size])
    self._encode, mask = tf.nn.max_pool_with_argmax(net, ksize=[FLAGS.pool_size, FLAGS.pool_size], strides=[1, 2, 2, 1],
                                           padding='SAME')
    # Decoder
    net = unpool(self._encode, mask)
    net = slim.conv2d_transpose(net, 16, [3, 3])
    net = slim.conv2d_transpose(net, 1, [5, 5])
    self._decode = net

    l2rec = tf.nn.l2_loss(slim.flatten(self._input) - slim.flatten(net))

    # Optimizer
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self._train = self._optimizer.minimize(l2rec)

  @ut.timeit
  def fetch_datasets(self):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(mnist.train)
    exit()
    pass

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
            feed_dict={self._input: batch[0]})
          # self._register_batch(loss, batch, encoding, reconstruction, step)
        # self._register_epoch(current_epoch, epochs_to_train, time.time() - start, sess)
      self._writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
      # meta = self._register_training()

  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    pass


if __name__ == '__main__':
  import sys

  model = TemporalPredictiveModel()
  model.train(FLAGS.max_epochs)
