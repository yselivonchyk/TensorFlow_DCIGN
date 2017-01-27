"""Doom AE with dropout. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import prettytensor as pt
import DropoutModel


FLAGS = tf.app.flags.FLAGS


class ConvModel(DropoutModel.DropoutModel):
  def __init__(self):
    super(ConvModel, self).__init__()
    self.model_id = 'conv'

  def encoder(self, input_tensor):
    print('Convolutional encoder')
    template = (pt.wrap(input_tensor).
                conv2d(5, 32, stride=2).
                conv2d(5, 64, stride=2).
                conv2d(5, 128, edges='VALID')
                .dropout(FLAGS.dropout).flatten()
            .fully_connected(self.layer_narrow))
    return template

  # def decoder(self, input):
  #   return (pt.wrap(input).
  #         reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
  #         deconv2d(3, 128, edges='VALID').
  #         deconv2d(5, 64, edges='VALID').
  #         deconv2d(5, 32, stride=2).
  #         deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
  #         flatten()).tensor

  def get_meta(self, meta=None):
    meta = super(ConvModel, self).get_meta()
    return meta

  # def load_meta(self, save_path):
  #   meta = super(ConvModel, self).load_meta()

if __name__ == '__main__':
  model = ConvModel()
  model.set_layer_sizes([500, 12, 500])
  model.train(100)
