import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network_utils as nut
import utils as ut
from Bunch import Bunch


INPUT = 'input'
FC = 'fully_connected'
CONV = 'convolutional'
POOL = 'max_pooling'
DO = 'dropout'
LOSS = 'loss'


activation_voc = {
  's': tf.nn.sigmoid,
  'r': tf.nn.relu,
  't': tf.nn.tanh,
  'i': None
}


CONFIG_COLOR = 30
PADDING = 'VALID'


def build_autoencoder(input, layer_config):
  layer_config = layer_config.split('-')
  layer_config = [parse_input(input)] + [parse(x) for x in layer_config]
  ut.print_info('Model config:', color=CONFIG_COLOR)
  enc = build_encoder(input, layer_config)
  dec = build_decoder(enc, layer_config)
  losses = build_losses(layer_config)
  return enc, dec, losses


def build_encoder(net, layer_config, i=1):
  if i == len(layer_config):
    return net

  cfg = layer_config[i]
  cfg.shape = net.get_shape().as_list()
  if cfg.type == FC:
    if len(cfg.shape) > 2:
      net = slim.flatten(net)
    net = slim.fully_connected(net, cfg.size, activation_fn=cfg.activation)
  elif cfg.type == CONV:
    net = slim.conv2d(net, cfg.size, [cfg.kernel, cfg.kernel],
                      activation_fn=cfg.activation, padding=PADDING)
  elif cfg.type == POOL:
    net = slim.max_pool2d(net, kernel_size=[cfg.kernel, cfg.kernel], stride=cfg.kernel)
  elif cfg.type == DO:
    print('dropout is decoder only')
  elif cfg.type == LOSS:
    cfg.arg1 = net
  elif cfg.type == INPUT:
    assert False
  ut.print_info('encoder_%d\t%s\t%s' % (i, str(cfg.shape), str(net)), color=CONFIG_COLOR)
  return build_encoder(net, layer_config, i + 1)


def build_decoder(net, layer_config, i=None):
  i = i if i is not None else len(layer_config) - 1

  cfg = layer_config[i]
  if cfg.type == FC:
    net = slim.fully_connected(net, int(np.prod(cfg.shape[1:])), activation_fn=cfg.activation)
    if layer_config[i-1].type != FC:
      net = tf.reshape(net, cfg.shape)
  elif cfg.type == CONV:
    net = slim.conv2d_transpose(net, cfg.shape[-1], [cfg.kernel, cfg.kernel],
                                activation_fn=cfg.activation, padding=PADDING)
  elif cfg.type == POOL:
    net = nut.upsample(net, cfg.kernel)
  elif cfg.type == DO:
    net = tf.nn.dropout(net, keep_prob=cfg.keep_prob)
  elif cfg.type == LOSS:
    cfg.arg2 = net
  elif cfg.type == INPUT:
    # if layer_config[1].type == FC:
    #   net = slim.fully_connected(net, int(np.prod(cfg.shape[1:])), activation_fn=tf.nn.relu)
    #   net = tf.reshape(net, cfg.shape)
    #   print('decoder_%d' % i, net)
    return net
  ut.print_info('decoder_%d \t%s' % (i, str(net)), color=CONFIG_COLOR)
  return build_decoder(net, layer_config, i-1)


def build_losses(layer_config):
  return []


def l2_loss(arg1, arg2, alpha=1.0, name='reco_loss'):
  error = slim.flatten(arg1) - slim.flatten(arg2)
  loss = tf.nn.l2_loss(error, name=name)
  return alpha * loss


def get_activation(descriptor):
  if 'c' not in descriptor and 'f' not in descriptor:
    return None, descriptor

  activation = tf.nn.relu if 'c' in descriptor else tf.nn.sigmoid
  if descriptor[-1] not in activation_voc.keys():
    return activation, descriptor
  return activation_voc[descriptor[-1]], descriptor[:-1]


def parse(descriptor):
  item = Bunch()

  if 'f' in descriptor:
    item.type = FC
    item.activation, descriptor = get_activation(descriptor)
    item.size = int(descriptor[1:])
  elif 'c' in descriptor:
    item.type = CONV
    item.activation, descriptor = get_activation(descriptor)
    params = descriptor.split('c')
    item.size = int(params[0])
    item.kernel = 3 if descriptor[-1] == 'c' else int(descriptor.split('c')[1])
  elif 'd' in descriptor:
    item.type = DO
    item.keep_prob = float(descriptor[1:])
  elif 'p' in descriptor:
    item.type = POOL
    item.kernel = int(descriptor[1:])
  elif 'l' in descriptor:
    item.type = LOSS
    item.loss_type = 'l2'
    item.alpha = float(descriptor.split('l')[0])
  else:
    print('What is "%s"? Check your writing 16c2i-7c-p3-0.01l-f10t-d0.3' % descriptor)
    assert False
  return item


def parse_input(input):
  item = Bunch()
  item.type = INPUT
  item.shape = input.get_shape().as_list()
  return item


if __name__ == '__main__':
  # build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '8c3-f10')
  # build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '8c-f12-f6')
  build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), 'f100-f10')
