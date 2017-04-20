import numpy as np
import tensorflow as tf
import autoencoder as ae
import os
import sys
import utils as ut
import visualization as vis
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS


# 1. load model
# 2. convert to static model
# 3. fetch data
# 4. feed one by one into the model -> create projection

def restore_flags(path):
  x = FLAGS.input_path
  log_file = os.path.join(path, '!note.txt')

  with open(log_file, 'r') as f:
    lines = f.readlines()

  # collect flags
  flag_stored = {}
  for l in lines:
    if ': \t' in l:
      parts = l[:-1].split(': \t')
      parts[0].strip()
      key = parts[0].strip()
      val = parts[1]
      flag_stored[key] = val

  print(flag_stored)
  print(FLAGS.__dict__['__flags'])

  flag_current = FLAGS.__dict__['__flags']
  for k in flag_stored.keys():
    if k in flag_current:
      # fix int issue '10.0' => '10'
      if type(flag_current[k]) == int:
        flag_stored[k] = flag_stored[k].split('.')[0]

      if type(flag_current[k]) == str:
        flag_stored[k] = flag_stored[k][1:-1]
      print(flag_stored[k], k)

      type_friendly_val = type(flag_current[k])(flag_stored[k])
      # print(k, type(dest[k]), flags[k], type(dest[k])(flags[k]))
      flag_current[k] = type_friendly_val


# def print_video(ecn, img, reco):




def data_to_img(img):
  h, w, c = img.shape
  base = np.zeros((h, 2*w, 3))
  base[:, :w, :] = img[:,:,:3]
  base[:, w:] = np.expand_dims(img[:,:,3], axis=2)
  return base


def _show_image(img, original=True):
  # index = 322 if original else 326
  index = (0, 1) if original else (4, 1)
  # ax = plt.subplot(index)
  ax = plt.subplot2grid((7, 2), index, rowspan=3)
  ax.imshow(img)
  ax.set_title('Original' if original else 'Reconstruction')
  ax.axis('off')


TAIL_LENGTH = 30


def animate(cur, enc, img, reco):
  fig = vis.get_figure()

  original = data_to_img(img[cur])
  reconstr = data_to_img(reco[cur])

  _show_image(original)
  _show_image(reconstr, original=False)

  # animation
  ax = plt.subplot2grid((7, 2), (0, 0), rowspan=7, projection='3d')
  ax.set_title('Trajectory')
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  ax.set_zticks([])

  if enc.shape[1] > 3:
    enc = enc[:, :4]

  # white
  data = enc[cur:]
  ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='w', s=1, zorder=15)
  # old
  tail_len = max(0, cur - TAIL_LENGTH)
  # print(tail_len)
  if tail_len > 0:
    data = enc[:tail_len]
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black', zorder=10, s=1,)
  # recent
  for i in range(0, -TAIL_LENGTH, -1):
    j = i + cur
    if j >= 0:
      # print(cur, i)
      data = enc[j]
      ax.scatter(data[0], data[1], data[2], c='b', s=(i+TAIL_LENGTH)/5, zorder=5)

  plt.show()

if __name__ == '__main__':
  path = os.getcwd()
  path = '/mnt/code/vd/TensorFlow_DCIGN/tmp/pred.16c3s2_32c3s2_32c3s2_16c3_f3__i_romb8.5.6'

  # write new flags: batch_size, path, save_path
  restore_flags(path)
  FLAGS.batch_size = 1

  path = FLAGS.input_path if len(FLAGS.test_path) == 0 else FLAGS.test_path
  path = '../../' + path
  path = sys.argv[-1] if len(sys.argv) == 3 else path
  FLAGS.test_path, FLAGS.input_path = path, path
  FLAGS.save_path = os.getcwd()
  FLAGS.model = 'ae'
  FLAGS.new_blur = False
  print(FLAGS.net, FLAGS.new_blur, FLAGS.test_path, FLAGS.input_path, os.getcwd())

  # run inference
  model = ae.Autoencoder(need_forlders=False)
  enc, reco = model.inference(max=100)
  img = model.test_set

  # enc, img, reco = np.arange(0, 364*3).reshape((364, 3)), np.random.rand(364, 80, 160, 4), np.random.rand(364, 80, 160, 4)

  for i in range(len(enc)):
    animate(i+100, enc, img, reco)

  # (364, 3)(364, 80, 160, 4)

