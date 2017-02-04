import visualization as vi
import utils as ut
import DoomModel as dm
import input as inp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from mpl_toolkits.mplot3d import Axes3D

# Next line to silence pyflakes. This import is needed.
Axes3D

FLAGS = tf.app.flags.FLAGS


def print_data(data, fig, subplot, is_3d=True):
  colors = np.arange(0, 180)
  colors = np.concatenate((colors, colors[::-1]))
  colors = vi._duplicate_array(colors, total_length=len(data))

  if is_3d:
    subplot = fig.add_subplot(subplot, projection='3d')
    subplot.set_title('All data')
    subplot.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=plt.cm.Spectral, picker=5)
  else:
    subsample = data[0:360] if len(data) < 2000 else data[0:720]
    subsample = np.concatenate((subsample, subsample))[0:len(subsample)+1]
    ut.print_info('subsample shape %s' % str(subsample.shape))
    subsample_colors = colors[0:len(subsample)]
    subplot = fig.add_subplot(subplot)
    subplot.set_title('First 360 elem')
    subplot.plot(subsample[:, 0], subsample[:, 1], picker=0)
    subplot.plot(subsample[0, 0], subsample[0, 1], picker=0)
    subplot.scatter(subsample[:, 0], subsample[:, 1], s=50, c=subsample_colors,
                    cmap=plt.cm.Spectral, picker=5)
  return subplot


class EncodingVisualizer:
  def __init__(self, fig, data):
    self.data = data
    self.fig = fig
    vi.visualize_encodings(data, grid=(3, 5), skip_every=5, fast=fast, fig=fig, interactive=True)
    plt.subplot(155).set_title(', '.join('hold on'))
    # fig.canvas.mpl_connect('button_press_event', self.on_click)
    fig.canvas.mpl_connect('pick_event', self.on_pick)
    try:
    # if True:
      ut.print_info('Checkpoint: %s' % FLAGS.load_from_checkpoint)
      self.model = dm.DoomModel()
      self.reconstructions = self.model.decode(data)
    except:
      ut.print_info("Model could not load from checkpoint %s" % str(sys.exc_info()), color=31)
      self.original_data, _ = inp.get_images(FLAGS.input_path)
      self.reconstructions = np.zeros(self.original_data.shape).astype(np.uint8)
    ut.print_info('INPUT: %s' % FLAGS.input_path.split('/')[-3])
    self.original_data, _ = inp.get_images(FLAGS.input_path)


  def on_pick(self, event):
    print(event)
    ind = event.ind
    print(ind)
    print(any([x for x in ind if x < 20]))
    orig = self.original_data[ind]
    reco = self.reconstructions[ind]
    column_picture, height = vi._stitch_images(orig, reco)
    picture = vi._reshape_column_image(column_picture, height, proportion=3)

    title = ''
    for i in range(len(ind)):
      title += ' ' + str(ind[i])
      if (i+1) % 8 == 0:
        title += '\n'
    plt.subplot(155).set_title(title)
    plt.subplot(155).imshow(picture)
    plt.show()

  def on_click(self, event):
    print('click', event)


def visualize_latest_from_visualization_folder(folder='./visualizations/', file=None):
  if file is None:
    file = ut.get_latest_file(folder, filter=r'.*\d+\.txt$')
    ut.print_info('Encoding file: %s' % file.split('/')[-1])
  data = np.loadtxt(file)  # [0:360]
  fig = plt.figure()
  vi.visualize_encodings(data, fast=fast, fig=fig,  interactive=True)
  fig.suptitle(file.split('/')[-1])
  fig.tight_layout()
  plt.show()


def visualize_from_checkpoint(checkpoint, epoch=None):
  assert os.path.exists(checkpoint)
  FLAGS.load_from_checkpoint = checkpoint
  file_filter = r'.*\d+\.txt$' if epoch is None else r'.*e\|%d.*' % epoch
  latest_file = ut.get_latest_file(folder=checkpoint, filter=file_filter)
  print(latest_file)
  ut.print_info('Encoding file: %s' % latest_file.split('/')[-1])
  data = np.loadtxt(latest_file)
  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 2)
  entity = EncodingVisualizer(fig, data)
  # fig.tight_layout()
  plt.show()


fast = True

if __name__ == '__main__':
  import sys



  path = sys.argv[1] if len(sys.argv) > 1 \
    else './tmp/ml__act|sigmoid__bs|30__h|500|10|500__init|na__inp|8pd3__lr|0.00003__opt|AO__seq|03'
  epoch = int(sys.argv[2]) if len(sys.argv) > 2 else None

  # path = './tmp/doom_bs__act|sigmoid__bs|30__h|500|12|500__init|na__inp|8pd3__lr|0.0004__opt|AO/'

  # import os
  # print('really? ', )

  if path is None:
    ut.print_info('Visualizing latest file from visualization folder')
    visualize_latest_from_visualization_folder()
    exit(0)

  is_embedding = '.txt' in path
  if is_embedding:
    ut.print_info('Visualizing encoding file')
    visualize_latest_from_visualization_folder(file=path)
    exit(0)

  is_checkpoint = '/tmp' in path
  if is_checkpoint:
    print('so', path)
    ut.print_info('Visualizing checkpoint data')
    visualize_from_checkpoint(checkpoint=path, epoch=epoch)
  else:
    ut.print_info('Visualizing latest from folder', color=34)
    visualize_latest_from_visualization_folder(folder=path)
