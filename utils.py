import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
import collections
import tensorflow as tf
import pickle
from scipy import misc
import re
import sys
import subprocess as sp
import warnings
import functools
# import scipy.stats as st



IMAGE_FOLDER = './img/'
TEMP_FOLDER = './tmp/'
EPOCH_THRESHOLD = 4
FLAGS = tf.app.flags.FLAGS

_start_time = None


# CONSOLE OPERATIONS


def reset_start_time():
  global _start_time
  _start_time = None


def _get_time_offset():
  global _start_time
  time = datetime.datetime.now()
  if _start_time is None:
    _start_time = time
    return '\t\t'
  sec = (time - _start_time).total_seconds()
  res = '(+%d)\t' % sec if sec < 60 else '(+%d:%02d)\t' % (sec/60, sec%60)
  return res


def print_time(*args, same_line=False):
  string = ''
  for a in args:
    string += str(a) + ' '
  time = datetime.datetime.now().time().strftime('%H:%M:%S')
  offset = _get_time_offset()
  res = '%s%s %s' % (str(time), offset, str(string))
  print_color(res, same_line=same_line)


def print_info(string, color=32, same_line=False):
  print_color('\t' + str(string), color=color, same_line=same_line)


same_line_prev = None


def print_color(string, color=33, same_line=False):
  global same_line_prev
  res = '%c[1;%dm%s%c[0m' % (27, color, str(string), 27)
  if same_line:
    print('\r                                                    ' +
          '                                                      ', end=' ')
    print('\r' + res, end=' ')
  else:
    # if same_line_prev:
    #   print('\n')
    print(res)
  same_line_prev = same_line


def mnist_select_n_classes(train_images, train_labels, num_classes, min=None, scale=1.0):
  result_images, result_labels = [], []
  for i, j in zip(train_images, train_labels):
    if np.sum(j[0:num_classes]) > 0:
      result_images.append(i)
      result_labels.append(j[0:num_classes])
  inputs = np.asarray(result_images)

  inputs *= scale
  if min is not None:
    inputs = inputs - np.min(inputs) + min
  return inputs, np.asarray(result_labels)


# IMAGE OPERATIONS


def _save_image(name='image', save_params=None, image=None):
  if save_params is not None and 'e' in save_params and save_params['e'] < EPOCH_THRESHOLD:
    print_info('IMAGE: output is not saved. epochs %d < %d' % (save_params['e'], EPOCH_THRESHOLD), color=31)
    return

  file_name = name if save_params is None else to_file_name(save_params)
  file_name += '.png'
  name = os.path.join(FLAGS.save_path, file_name)

  if image is not None:
    misc.imsave(name, arr=image, format='png')


def _show_picture(pic):
  fig = plt.figure()
  size = fig.get_size_inches()
  fig.set_size_inches(size[0], size[1] * 2, forward=True)
  plt.imshow(pic, cmap='Greys_r')


def concat_images(im1, im2, axis=0):
  if im1 is None:
    return im2
  return np.concatenate((im1, im2), axis=axis)


def _reconstruct_picture_line(pictures, shape):
  line_picture = None
  for _, img in enumerate(pictures):
    if len(img.shape) == 1:
      img = (np.reshape(img, shape))
    if len(img.shape) == 3 and img.shape[2] == 1:
      img = (np.reshape(img, (img.shape[0], img.shape[1])))
    line_picture = concat_images(line_picture, img)
  return line_picture


def show_plt():
  plt.show()


def _construct_img_shape(img):
  assert int(np.sqrt(img.shape[0])) == np.sqrt(img.shape[0])
  return int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0])), 1


def images_to_uint8(func):
  def normalize(arr):
    # if type(arr) == np.ndarray and arr.dtype != np.uint8 and len(arr.shape) >= 3:
    if type(arr) == np.ndarray and len(arr.shape) >= 3:
      if np.min(arr) < 0:
        print('image array normalization: negative values')
      if np.max(arr) < 10:
        arr *= 255
      if arr.shape[-1] == 4 or arr.shape[-1] == 2:
        old_shape = arr.shape
        arr = arr[..., :arr.shape[-1]-1]
      return arr.astype(np.uint8)
    return arr

  def func_wrapper(*args, **kwargs):
    new_args = [normalize(el) for el in args]
    new_kwargs = {k: normalize(kwargs[k]) for _, k in enumerate(kwargs)}
    return func(*tuple(new_args), **new_kwargs)
  return func_wrapper


@images_to_uint8
def reconstruct_images_epochs(epochs, original=None, save_params=None, img_shape=None):
  full_picture = None
  img_shape = img_shape if img_shape is not None else _construct_img_shape(epochs[0][0])

  # print(original.dtype, epochs.dtype, np.max(original), np.max(epochs))

  if original.dtype != np.uint8:
    original = (original * 255).astype(np.uint8)
  if epochs.dtype != np.uint8:
    epochs = (epochs * 255).astype(np.uint8)

  # print('image reconstruction: ', original.dtype, epochs.dtype, np.max(original), np.max(epochs))

  if original is not None and epochs is not None and len(epochs) >= 3:
    min_ref, max_ref = np.min(original), np.max(original)
    print_info('epoch avg: (original: %s) -> %s' % (
    str(np.mean(original)), str((np.mean(epochs[0]), np.mean(epochs[1]), np.mean(epochs[2])))))
    print_info('reconstruction char. in epochs (min, max)|original: (%f %f)|(%f %f)' % (
    np.min(epochs[1:]), np.max(epochs), min_ref, max_ref))

  if epochs is not None:
    for _, epoch in enumerate(epochs):
      full_picture = concat_images(full_picture, _reconstruct_picture_line(epoch, img_shape), axis=1)
  if original is not None:
    full_picture = concat_images(full_picture, _reconstruct_picture_line(original, img_shape), axis=1)
  _show_picture(full_picture)
  _save_image(save_params=save_params, image=full_picture)


def model_to_file_name(FLAGS, folder=None, ext=None):
  name = '%s.%s__i_%s' % (FLAGS.model_type, FLAGS.net.replace('-', '_'), FLAGS.input_name)
  if ext:
    name += '.' + ext
  if folder:
    name = os.path.join(folder, name)
  return name


def mkdir(folders):
  if isinstance(folders, str):
    folders = [folders]
  for _, folder in enumerate(folders):
    if not os.path.exists(folder):
      os.mkdir(folder)


def configure_folders(FLAGS):
  folder_name = model_to_file_name(FLAGS) + '/'
  FLAGS.save_path = os.path.join(TEMP_FOLDER, folder_name)
  FLAGS.logdir = os.path.join(FLAGS.save_path, 'log')
  mkdir([TEMP_FOLDER, IMAGE_FOLDER, FLAGS.save_path, FLAGS.logdir])


def get_latest_file(folder="./visualizations/", filter=None):
  latest_file, latest_mod_time = None, None
  for root, dirs, files in os.walk(folder):
    # print(root, dirs, files)
    if filter:
      files = [x for x in files if re.match(filter, x)]
    # print('\n\r'.join(files))
    for file in files:
      if '.txt' in file:
        file_path = os.path.join(root, file)
        modification_time = os.path.getmtime(file_path)
        if not latest_mod_time or modification_time > latest_mod_time:
          latest_mod_time = modification_time
          latest_file = file_path
  if latest_file is None:
    print_info('Could not find file matching %s' % str(filter))
  return latest_file


def concatenate(x, y, take=None):
  """
  Stitches two np arrays together until maximum length, when specified
  """
  if take is not None and x is not None and len(x) >= take:
    return x
  if x is None:
    res = y
  else:
    res = np.concatenate((x, y))
  return res[:take] if take is not None else res


# MISC


def list_object_attributes(obj):
  print('Object type: %s\t\tattributes:' % str(type(obj)))
  print('\n\t'.join(map(str, obj.__dict__.keys())))


def print_list(list):
  print('\n'.join(map(str, list)))


def print_float_list(list, format='%.4f'):
  return ' '.join(map(lambda x: format%x, list))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r %2.2f sec' % (method.__name__, te-ts))
        return result
    return timed


def deprecated(func):
  '''This is a decorator which can be used to mark functions
  as deprecated. It will result in a warning being emitted
  when the function is used.'''

  @functools.wraps(func)
  def new_func(*args, **kwargs):
    warnings.warn_explicit(
      "Call to deprecated function {}.".format(func.__name__),
      category=DeprecationWarning,
      filename=func.func_code.co_filename,
      lineno=func.func_code.co_firstlineno + 1
    )
    return func(*args, **kwargs)

  return new_func


import numpy as np
ACCEPTABLE_AVAILABLE_MEMORY = 1024


def mask_busy_gpus(leave_unmasked=1, random=True):
  try:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) < leave_unmasked:
      print('Found only %d usable GPUs in the system' % len(available_gpus))
      exit(0)

    if random:
      available_gpus = np.asarray(available_gpus)
      np.random.shuffle(available_gpus)

    # update CUDA variable
    gpus = available_gpus[:leave_unmasked]
    setting = ','.join(map(str, gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = setting
    print('Left next %d GPU(s) unmasked: [%s] (from %s available)'
          % (leave_unmasked, setting, str(available_gpus)))
  except FileNotFoundError as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked')
    print(e)
  except sp.CalledProcessError as e:
    print("Error on GPU masking:\n", e.output)


def _output_to_list(output):
  return output.decode('ascii').split('\n')[:-1]


def get_gpu_free_session(memory_fraction=0.1):
  import tensorflow as tf
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
  return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def parse_params():
  params = {}
  for i, param in enumerate(sys.argv):
    if '-' in param:
      params[param[1:]] = sys.argv[i+1]
  print(params)
  return params


def print_flags(FLAGS):
  x = FLAGS.input_path
  res = str(FLAGS.__dict__['__flags'])
  res = res.replace(', ', '\n')
  print_info(res)


# def _build_gaussian_kernel(k_size, nsig, channels):
#   interval = (2 * nsig + 1.) / k_size
#   x = np.linspace(-nsig - interval / 2., nsig + interval / 2., k_size + 1)
#   kern1d = np.diff(st.norm.cdf(x))
#   kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#   kernel = kernel_raw / kernel_raw.sum()
#   out_filter = np.array(kernel, dtype=np.float32)
#   out_filter = out_filter.reshape((k_size, k_size, 1, 1))
#   out_filter = np.repeat(out_filter, channels, axis=2)
#   return out_filter
#
#
# def blur_gaussian(input, sigma, filter_size):
#   num_channels = input.get_shape().as_list()[3]
#   with tf.variable_scope('gaussian_filter'):
#     kernel = _build_gaussian_kernel(filter_size, sigma, num_channels)
#     kernel = tf.Variable(tf.convert_to_tensor(kernel), name='gauss_weight')
#     output = tf.nn.depthwise_conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
#     return output, kernel


if __name__ == '__main__':
  data = []
  for i in range(10):
      data.append((str(i), np.random.rand(1000)))
  plot_epoch_progress({'f': 'test'}, data, True)


mask_busy_gpus()