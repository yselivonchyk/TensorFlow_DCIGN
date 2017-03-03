import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tools.checkpoint_utils as ch_utils


# POOLING


def max_pool_with_argmax(net, stride):
  """
  Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
  Therefore, we use max_pool_with_argmax to extract mask and
  plain max_pool for, eeem... max_pooling.
  """
  with tf.name_scope('MaxPoolArgMax'):
    _, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=[1, stride, stride, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')
    mask = tf.stop_gradient(mask)
    net = slim.max_pool2d(net, kernel_size=[stride, stride],  stride=FLAGS.pool_size)
    return net, mask


# Thank you, @https://github.com/Pepslee
def unpool(net, mask, stride):
  assert mask is not None
  with tf.name_scope('UnPool2D'):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
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
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


def upsample(net, stride, mode='ZEROS'):
  """
  Imitate reverse operation of Max-Pooling by either placing original max values
  into a fixed postion of upsampled cell:
  [0.9] =>[[.9, 0],   (stride=2)
           [ 0, 0]]
  or copying the value into each cell:
  [0.9] =>[[.9, .9],  (stride=2)
           [ .9, .9]]

  :param net: 4D input tensor with [batch_size, width, heights, channels] axis
  :param stride:
  :param mode: string 'ZEROS' or 'COPY' indicating which value to use for undefined cells
  :return:  4D tensor of size [batch_size, width*stride, heights*stride, channels]
  """
  assert mode in ['COPY', 'ZEROS']
  with tf.name_scope('Upsampling'):
    net = _upsample_along_axis(net, 2, stride, mode=mode)
    net = _upsample_along_axis(net, 1, stride, mode=mode)
    return net


def _upsample_along_axis(volume, axis, stride, mode='ZEROS'):
  shape = volume.get_shape().as_list()

  assert mode in ['COPY', 'ZEROS']
  assert 0 <= axis < len(shape)

  target_shape = shape[:]
  target_shape[axis] *= stride

  padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
  parts = [volume] + [padding for _ in range(stride - 1)]
  volume = tf.concat(parts, min(axis+1, len(shape)-1))

  volume = tf.reshape(volume, target_shape)
  return volume


# VARIABLES


def print_model_info(trainable=False):
  if not trainable:
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      print(v.name, v.get_shape())
  else:
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      value = v.eval()
      print('TRAINABLE_VARIABLES', v.name, v.get_shape(), 'm:%.4f v:%.4f' % (value.mean(), value.std()))


def list_checkpoint_vars(folder):
  f = ch_utils.list_variables(folder)
  print('\n'.join(map(str, f)))