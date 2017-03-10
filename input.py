import os
import numpy as np
import utils as ut
import json
import scipy.ndimage.filters as filters
import time
from PIL import Image
import tarfile
import io


INPUT_FOLDER = '../data/circle_basic_1/img/32_32'


def _is_combination_of_image_depth(folder):
  return '/dep' not in folder and '/img' not in folder


def get_action_data(folder):
  folder = folder.replace('/tmp_grey', '')
  folder = folder.replace('/tmp', '')
  folder = folder.replace('/img', '')
  folder = folder.replace('/dep', '')
  file = os.path.join(folder, 'action.txt')
  if not os.path.exists(file):
    return np.asarray([])
  action_data = json.load(open(file, 'r'))[:]
  # print(action_data)
  res = []
  # for i, action in enumerate(action_data):
  #   print(action)
  #   res.append(
  #     (
  #       action[0],
  #       action[1],
  #       action[2][3] or action[2][4] or action[2][5] or action[2][6],
  #       action[2][18] != 0
  #     )
  #   )
  # print([print(x[0]) for x in action_data[0:10]])
  res = [x[3][:2] for x in action_data]
  return np.abs(np.asarray(res))


def read_ds_zip(path):
  dep, img = {}, {}
  tar = tarfile.open(path, "r:gz")

  for member in tar.getmembers():
    if '.jpg' not in member.name or not ('/dep/' in member.name or '/img/' in member.name):
      # print('skipped', member)
      continue
    collection = dep if '/dep/' in member.name else img
    index = int(member.name.split('/')[-1][1:-4])
    f = tar.extractfile(member)
    if f is not None:
      content = f.read()
      image = Image.open(io.BytesIO(content))
      collection[index] = np.array(image)
  assert len(img) == len(dep)

  shape = [len(img)] + list(img[index].shape)
  shape[-1] += 1

  dataset = np.zeros(shape, np.uint8)
  for i, k in enumerate(sorted(img)):
    dataset[i, ..., :-1] = img[k]
    dataset[i, ..., -1] = dep[k]
  return dataset#, shape[1:]


def get_shape_zip(path):
  tar = tarfile.open(path, "r:gz")
  for member in tar.getmembers():
    if '.jpg' not in member.name or '/img/' not in member.name:
      continue
    f = tar.extractfile(member)
    content = f.read()
    image = Image.open(io.BytesIO(content))
    shape = list(np.array(image).shape)
    shape[-1] += 1
    return shape


def rescale_ds(ds, min, max):
  ut.print_info('rescale call: (min: %s, max: %s) %d' % (str(min), str(max), len(ds)))
  if max is None:
    return np.asarray(ds) - np.min(ds)
  ds_min, ds_max = np.min(ds), np.max(ds)
  ds_gap = ds_max - ds_min
  scale_factor = (max - min) / ds_gap
  ds = np.asarray(ds) * scale_factor
  shift_factor = min - np.min(ds)
  ds += shift_factor
  return ds


def get_input_name(input_folder):
  spliter = '/img/' if '/img/' in input_folder else '/dep/'
  main_part = input_folder.split(spliter)[0]
  name = main_part.split('/')[-1]
  name = name.replace('.tar.gz', '')
  ut.print_info('input folder: %s -> %s' % (input_folder.split('/'), name))
  return name


def permute_array(array, random_state=None):
  return permute_data((array,))[0]


def permute_data(arrays, random_state=None):
  """Permute multiple numpy arrays with the same order."""
  if any(len(a) != len(arrays[0]) for a in arrays):
    raise ValueError('All arrays must be the same length.')
  if not random_state:
    random_state = np.random
  order = random_state.permutation(len(arrays[0]))
  return [a[order] for a in arrays]


def apply_gaussian(images, sigma):
  if sigma == 0:
    return images

  res = images.copy()
  for i, image in enumerate(res):
    for channel in range(image.shape[-1]):
      image[:, :, channel] = filters.gaussian_filter(image[:, :, channel], sigma)
  ut.print_info('blur s:%.1f[%.1f>%.1f]' % (sigma, images[2,10,10,0], res[2,10,10,0]))
  return res


def permute_array_in_series(array, series_length, allow_shift=True):
  res, permutation = permute_data_in_series((array,), series_length)
  return res[0], permutation


def permute_data_in_series(arrays, series_length, allow_shift=True):
  shift_possibilities = len(arrays[0]) % series_length
  series_count = int(len(arrays[0]) / series_length)

  shift = 0
  if allow_shift:
    if shift_possibilities == 0:
      shift_possibilities += series_length
      series_count -= 1
    shift = np.random.randint(0, shift_possibilities+1, 1, dtype=np.int32)[0]

  series = np.arange(0, series_count * series_length)\
    .astype(np.int32)\
    .reshape((series_count, series_length))

  series = np.random.permutation(series)
  data_permutation = series.reshape((series_count * series_length))
  data_permutation += shift

  remaining_elements = np.arange(0, len(arrays[0])).astype(np.int32)
  remaining_elements = np.delete(remaining_elements, data_permutation)
  data_permutation = np.concatenate((data_permutation, remaining_elements))

  # print('assert', len(arrays[0]), len(data_permutation))
  assert len(data_permutation) == len(arrays[0])
  return [a[data_permutation] for a in arrays], data_permutation


def pad_set(set, batch_size):
  length = len(set)
  if length % batch_size == 0:
    return set
  padding_len = batch_size - length % batch_size
  if padding_len != 0:
    pass
    # ut.print_info('Non-zero padding: %d' % padding_len, color=31)
  # print('pad set', set.shape, select_random(padding_len, set=set).shape)
  return np.concatenate((set, select_random(padding_len, set=set)))


def select_random(n, length=None, set=None):
  assert length is None or set is None
  length = length if set is None else len(set)
  select = np.random.permutation(np.arange(length, dtype=np.int))[:n]
  if set is None:
    return select
  else:
    return set[select]

if __name__ == '__main__':
  print(read_ds_zip('/home/eugene/repo/data/tmp/romb8.5.6.tar.gz').shape)