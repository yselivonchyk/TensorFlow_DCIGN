import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network_utils as nut
import utils as ut
import re
import os
from Bunch import Bunch
from model_interpreter import *


def _log_graph():
  path = '/tmp/interpreter'
  with tf.Session() as sess:
    tf.global_variables_initializer()
    tf.summary.FileWriter(path, sess.graph)
    ut.print_color(os.path.abspath(path), color=33)


def _test_parameter_reuse_conv():
  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3s2-16c3s2-32c3s2-1c3-f4')

  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3s2-16c3s2-32c3s2-1c3-f4')
  l2_loss(input, model.decode)

  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, model.config)
  l2_loss(input, model.decode)


def _test_parameter_reuse_decoder():
  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3s2-16c3s2-32c3s2-16c3-f4')

  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3s2-16c3s2-32c3s2-16c3-f4')

  input_enc = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input_encoder')
  encoder = build_encoder(input_enc, model.config, reuse=True)

  input_dec = tf.placeholder(tf.float32, (2, 4), name='input_decoder')
  decoder = build_decoder(input_dec, model.config, reuse=True)

  l2_loss(input, model.decode)
  l2_loss(encoder, model.encode)
  l2_loss(decoder, model.decode)


def _test_armgax_ae():
  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3-ap2-16c3-ap2-32c3-ap2-16c3-f4')

  input = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input')
  model = build_autoencoder(input, '8c3-ap2-16c3-ap2-32c3-ap2-16c3-f4')

  input_enc = tf.placeholder(tf.float32, (2, 120, 120, 4), name='input_encoder')
  encoder = build_encoder(input_enc, model.config, reuse=True)

  input_dec = tf.placeholder(tf.float32, (2, 4), name='input_decoder')
  decoder = build_decoder(input_dec, model.config, reuse=True)

  l2_loss(input, model.decode)
  l2_loss(encoder, model.encode)
  l2_loss(decoder, model.decode)


def _test_multiple_decoders_unpool_wiring():
  input = tf.placeholder(tf.float32, (2, 16, 16, 3), name='input')
  model1 = build_autoencoder(input, '8c3-ap2-f4')
  model2 = build_autoencoder(input,  model1.config)
  enc_1 = tf.placeholder(tf.float32, (2, 4), name='enc_1')
  enc_2 = tf.placeholder(tf.float32, (2, 4), name='enc_2')
  decoder1 = build_decoder(enc_1, model1.config, reuse=True, masks=model1.mask_list)
  decoder2 = build_decoder(enc_2, model2.config, reuse=True, masks=model2.mask_list)


def _visualize_models():
  input = tf.placeholder(tf.float32, (128, 160, 120, 4), name='input_fc')
  model = build_autoencoder(input, 'f100-f3')
  loss= l2_loss(input, model.decode, name='Loss_reconstruction_FC')

  input = tf.placeholder(tf.float32, (128, 160, 120, 4), name='input_conv')
  model = build_autoencoder(input, '16c3s2-32c3s2-32c3s2-16c3-f3')
  loss= l2_loss(input, model.decode,  name='Loss_reconstruction_conv')

  input = tf.placeholder(tf.float32, (128, 160, 120, 4), name='input_wwae')
  model = build_autoencoder(input, '16c3-ap2-32c3-ap2-16c3-f3')
  loss= l2_loss(input, model.decode, name='Loss_reconstruction_WWAE')


if __name__ == '__main__':
  # print(re.match('\d+c\d+(s\d+)?[r|s|i|t]?', '8c3s2'))
  # model = build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '8c3s2-16c3s2-30c3s2-16c3-f4')
  # _test_multiple_decoders_unpool_wiring()
  _visualize_models()

  # build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '10c3-f100-f10')
  # _test_parameter_reuse_conv()
  # _test_parameter_reuse_decoder()
  # _test_armgax_ae()
  _log_graph()
