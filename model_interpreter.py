import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network_utils as nut
import utils as ut
import re
from Bunch import Bunch

INPUT = 'input'
FC = 'fully_connected'
CONV = 'convolutional'
POOL = 'max_pooling'
POOL_ARG = 'maxpool_with_args'
DO = 'dropout'
LOSS = 'loss'

activation_voc = {
  's': tf.nn.sigmoid,
  'r': tf.nn.relu,
  't': tf.nn.tanh,
  'i': None
}

CONFIG_COLOR = 30
PADDING = 'SAME'


def clean_unpooling_masks(layer_config):
  for cfg in layer_config:
    if cfg.type == POOL_ARG:
      cfg.argmax = None


def build_autoencoder(input, layer_config):
  reuse_model = isinstance(layer_config, list)
  if not reuse_model:
    layer_config = layer_config.split('-')
    layer_config = [parse_input(input)] + [parse(x) for x in layer_config]
  ut.print_info('Model config:', color=CONFIG_COLOR)
  enc = build_encoder(input, layer_config, reuse=reuse_model)
  dec = build_decoder(enc, layer_config, reuse=reuse_model)
  clean_unpooling_masks(layer_config)
  losses = build_losses(layer_config)
  return Bunch(encode=enc, decode=dec, losses=losses, config=layer_config)


def build_encoder(net, layer_config, i=1, reuse=False):
  if i == len(layer_config):
    return net

  cfg = layer_config[i]
  cfg.shape = net.get_shape().as_list()
  name = cfg.enc_op_name if reuse else None
  if cfg.type == FC:
    if len(cfg.shape) > 2:
      net = slim.flatten(net)
    net = slim.fully_connected(net, cfg.size, activation_fn=cfg.activation,
                               scope=name, reuse=reuse)
  elif cfg.type == CONV:
    net = slim.conv2d(net, cfg.size, [cfg.kernel, cfg.kernel], stride=cfg.stride,
                      activation_fn=cfg.activation, padding=PADDING,
                      scope=name, reuse=reuse)
  elif cfg.type == POOL_ARG:
    net, cfg.argmax = nut.max_pool_with_argmax(net, cfg.kernel)
    # if not reuse:
    #   mask = nut.fake_arg_max_of_max_pool(cfg.shape, cfg.kernel)
    #   cfg.argmax_dummy = tf.constant(mask.flatten(), shape=mask.shape)
  elif cfg.type == POOL:
    net = slim.max_pool2d(net, kernel_size=[cfg.kernel, cfg.kernel], stride=cfg.kernel)
  elif cfg.type == DO:
    print('dropout is decoder only')
  elif cfg.type == LOSS:
    cfg.arg1 = net
  elif cfg.type == INPUT:
    assert False

  if not reuse:
    cfg.enc_op_name = net.name.split('/')[0]
  ut.print_info('\rencoder_%d\t%s\t%s' % (i, str(net), cfg.enc_op_name), color=CONFIG_COLOR)
  return build_encoder(net, layer_config, i + 1, reuse=reuse)


def build_decoder(net, layer_config, i=None, reuse=False):
  i = i if i is not None else len(layer_config) - 1

  cfg = layer_config[i]
  name = cfg.dec_op_name if reuse else None

  if len(layer_config) > i + 1:
    if len(layer_config[i + 1].shape) != len(net.get_shape().as_list()):
      net = tf.reshape(net, layer_config[i + 1].shape)

  if cfg.type == FC:
    net = slim.fully_connected(net, int(np.prod(cfg.shape[1:])), scope=name,
                               activation_fn=cfg.activation, reuse=reuse)
  elif cfg.type == CONV:
    net = slim.conv2d_transpose(net, cfg.shape[-1], [cfg.kernel, cfg.kernel], stride=cfg.stride,
                                activation_fn=cfg.activation, padding=PADDING,
                                scope=name, reuse=reuse)
  elif cfg.type == POOL_ARG:
    if cfg.argmax is not None:
      net = nut.unpool(net, mask=cfg.argmax, stride=cfg.kernel)
    else:
      net = nut.upsample(net, stride=cfg.kernel)
  elif cfg.type == POOL:
    net = nut.upsample(net, cfg.kernel)
  elif cfg.type == DO:
    net = tf.nn.dropout(net, keep_prob=cfg.keep_prob)
  elif cfg.type == LOSS:
    cfg.arg2 = net
  elif cfg.type == INPUT:
    return net
  if not reuse:
    cfg.dec_op_name = net.name.split('/')[0]
  ut.print_info('\rdecoder_%d \t%s' % (i, str(net)), color=CONFIG_COLOR)
  return build_decoder(net, layer_config, i - 1, reuse=reuse)


def build_losses(layer_config):
  return []


def l2_loss(arg1, arg2, alpha=1.0, name='reco_loss'):
  with tf.name_scope('L2_loss'):
    loss = tf.nn.l2_loss(arg1 - arg2, name=name)
    return alpha * loss


def get_activation(descriptor):
  if 'c' not in descriptor and 'f' not in descriptor:
    return None
  activation = tf.nn.relu if 'c' in descriptor else tf.nn.sigmoid
  act_descriptor = re.search('[r|s|i|t]&', descriptor)
  if act_descriptor is None:
    return activation
  act_descriptor = act_descriptor.group(0)
  return activation_voc[act_descriptor]


def _get_cfg_dummy():
  return Bunch(enc_op_name=None, dec_op_name=None)


def parse(descriptor):
  item = _get_cfg_dummy()

  match = re.match(r'^((\d+c\d+(s\d+)?[r|s|i|t]?)'
                   r'|(f\d+[r|s|i|t]?)'
                   r'|(d0?\.?[\d+]?)'
                   r'|(d0?\.?[\d+]?)'
                   r'|(p\d+)'
                   r'|(ap\d+))$', descriptor)
  assert match is not None, 'Check your writing: %s (f10i-3c64r-d0.1-p2-ap2)' % descriptor


  if 'f' in descriptor:
    item.type = FC
    item.activation = get_activation(descriptor)
    item.size = int(re.search('f\d+', descriptor).group(0)[1:])
  elif 'c' in descriptor:
    item.type = CONV
    item.activation = get_activation(descriptor)
    item.kernel = int(re.search('c\d+', descriptor).group(0)[1:])
    stride = re.search('s\d+', descriptor)
    item.stride = int(stride.group(0)[1:]) if stride is not None else 1
    item.size = int(re.search('\d+c', descriptor).group(0)[:-1])
  elif 'd' in descriptor:
    item.type = DO
    item.keep_prob = float(descriptor[1:])
  elif 'ap' in descriptor:
    item.type = POOL_ARG
    item.kernel = int(descriptor[2:])
  elif 'p' in descriptor:
    item.type = POOL
    item.kernel = int(descriptor[1:])
  elif 'l' in descriptor:
    item.type = LOSS
    item.loss_type = 'l2'
    item.alpha = float(descriptor.split('l')[0])
  else:
    print('What is "%s"? Check your writing 16c2i-7c3r-p3-0.01l-f10t-d0.3' % descriptor)
    assert False
  return item


def parse_input(input):
  item = _get_cfg_dummy()
  item.type = INPUT
  item.shape = input.get_shape().as_list()
  return item


def _log_graph():
  with tf.Session() as sess:
    tf.global_variables_initializer()
    tf.summary.FileWriter('/tmp/interpreter', sess.graph)


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



if __name__ == '__main__':
  # print(re.match('\d+c\d+(s\d+)?[r|s|i|t]?', '8c3s2'))
  model = build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '8c3s2-16c3s2-30c3s2-16c3-f4')
  # build_autoencoder(tf.placeholder(tf.float32, (2, 16, 16, 3), name='input'), '10c3-f100-f10')
  # _test_parameter_reuse_conv()
  # _test_parameter_reuse_decoder()
  # _test_armgax_ae()
  _log_graph()
