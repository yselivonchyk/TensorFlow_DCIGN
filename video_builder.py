import numpy as np
import tensorflow as tf
import autoencoder as ae
import os
import sys

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





if __name__ == '__main__':
  path = os.getcwd()
  # path = '/mnt/code/vd/TensorFlow_DCIGN/tmp/pred.16c3s2_32c3s2_32c3s2_16c3_f3__i_romb8.5.6'


  restore_flags(path)
  # write new flags: batch_size, path, save_path
  FLAGS.batch_size = 1

  path = FLAGS.input_path if len(FLAGS.test_path) == 0 else FLAGS.test_path
  path = '../../' + path
  path = sys.argv[-1] if len(sys.argv) == 3 else path
  FLAGS.test_path, FLAGS.input_path = path, path
  print(FLAGS.test_path, FLAGS.input_path, os.getcwd())

  FLAGS.save_path = os.getcwd()
  FLAGS.model = 'ae'
  FLAGS.new_blur = False

  print(FLAGS.net, FLAGS.new_blur)


  # run infirence
  model = ae.Autoencoder(need_forlders=False)
  enc, img = model.inference(max=100)
  # (364, 3)(364, 80, 160, 4)



