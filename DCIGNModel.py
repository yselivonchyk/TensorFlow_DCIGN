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
import prettytensor.bookkeeper as bookkeeper
import deconv
from tensorflow.python.ops import gradients
from prettytensor.tutorial import data_utils
import IGNModel

FLAGS = tf.app.flags.FLAGS

DEV = False


class DCIGNModel(IGNModel.IGNModel):
  model_id = 'dcign'

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


def parse_params():
  params = {}
  for i, param in enumerate(sys.argv):
    if '-' in param:
      params[param[1:]] = sys.argv[i+1]
  print(params)
  return params


if __name__ == '__main__':
  epochs = 500
  import sys

  FLAGS.save_every = 5
  FLAGS.save_encodings_every = 2


  model = DCIGNModel()
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
