import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tools.checkpoint_utils as ch_utils
import scipy.stats as st
import inspect

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
    net = slim.max_pool2d(net, kernel_size=[stride, stride],  stride=stride)
    return net, mask


def fake_arg_max_of_max_pool(shape, stride=2):
  assert shape[1] % stride == 0 and shape[2] % stride == 0, \
    'Smart padding is not supported. Indexes %s are not multiple of stride:%d' % (str(shape[1:3]), stride)
  mask = np.arange(np.prod(shape[1:]))
  mask = mask.reshape(shape[1:])
  mask = mask[::stride, ::stride, :]
  mask = np.tile(mask, (shape[0], 1, 1, 1))
  return mask


# Thank you, @https://github.com/Pepslee
def unpool(net, mask, stride=2):
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
  assert list(inspect.signature(tf.concat).parameters.items())[1][0] == 'axis', 'Wrong TF version'
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
  # print(folder)
  f = ch_utils.list_variables(folder)
  # print(f)
  print('\n'.join(map(str, f)))


def get_variable(checkpoint, name):
  var = ch_utils.load_variable(tf.train.latest_checkpoint(checkpoint), name)
  return var


def scope_wrapper(scope_name):
  def scope_decorator(func):
    def func_wrapper(*args, **kwargs):
      with tf.name_scope(scope_name):
        return func(*args, **kwargs)
    return func_wrapper
  return scope_decorator


# Gaussian blur


def _build_gaussian_kernel(k_size, nsig, channels):
  interval = (2 * nsig + 1.) / k_size
  x = np.linspace(-nsig - interval / 2., nsig + interval / 2., k_size + 1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw / kernel_raw.sum()
  out_filter = np.array(kernel, dtype=np.float32)
  out_filter = out_filter.reshape((k_size, k_size, 1, 1))
  out_filter = np.repeat(out_filter, channels, axis=2)
  return out_filter


def blur_gaussian(input, sigma, filter_size):
  num_channels = input.get_shape().as_list()[3]
  with tf.variable_scope('gaussian_filter'):
    kernel = _build_gaussian_kernel(filter_size, sigma, num_channels)
    kernel = tf.constant(kernel.flatten(), shape=kernel.shape, name='gauss_weight')
    output = tf.nn.depthwise_conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    return output, kernel