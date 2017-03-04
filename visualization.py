from sklearn.manifold import TSNE
import sklearn.manifold as mn
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pw
import scipy.spatial.distance as dist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import utils as ut
import time
import tensorflow as tf

# Next line to silence pyflakes. This import is needed.
Axes3D

# colors = ['grey', 'red', 'magenta']

FLAGS = tf.app.flags.FLAGS
COLOR_MAP = plt.cm.Spectral
PICKER_SENSITIVITY = 5


def scatter(plot, data, is3d, colors):
  if is3d:
    plot.scatter(data[:, 0], data[:, 1], data[:, 2],
                 marker='.',
                 c=colors,
                 cmap=plt.cm.Spectral,
                 picker=PICKER_SENSITIVITY)
  else:
    plot.scatter(data[:, 0], data[:, 1],
                 c=colors,
                 cmap=plt.cm.Spectral,
                 picker=PICKER_SENSITIVITY)


def print_data_only(data, file_name, fig=None, interactive=False):
  fig = fig if fig is not None else plt.figure()
  subplot_number = 121 if fig is not None else 111
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 1)

  colors = _build_radial_colors(len(data))
  if data.shape[1] > 2:
    subplot = plt.subplot(subplot_number, projection='3d')
    subplot.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors,
                    cmap=COLOR_MAP, picker=PICKER_SENSITIVITY)
    subplot.plot(data[:, 0], data[:, 1], data[:, 2])
  else:
    subplot = plt.subplot(subplot_number)
    subplot.scatter(data[:, 0], data[:, 1], c=colors,
                    cmap=COLOR_MAP, picker=PICKER_SENSITIVITY)
  if not interactive:
    save_fig(file_name, fig)


def create_gif_from_folder(folder):
  # fig = plt.figure()
  # gif_folder = os.path.join(FLAGS.save_path, 'gif')
  # if not os.path.exists(gif_folder):
  #   os.mkdir(gif_folder)
  # epoch = file_name.split('_e|')[-1].split('_')[0]
  # gif_path = os.path.join(gif_folder, epoch)
  # subplot = plt.subplot(111, projection='3d')
  # subplot.scatter(data[0], data[1], data[2], c=colors, cmap=color_map)
  # save_fig(file_name)
  pass


def manual_pca(data, std_threshold=0.01, num_threshold=3):
  """remove meaningless dimensions"""
  std = data[0:300].std(axis=0)

  order = np.argsort(std)[::-1]
  # order = np.arange(0, data.shape[1]).astype(np.int32)
  std = std[order]
  # filter components by STD but take at least 3

  meaningless = [order[i] for i, x in enumerate(std) if x <= std_threshold]
  if any(meaningless) and data.shape[1] > 3:
    ut.print_info('meaningless dimensions on visualization: %s' % str(meaningless))

  order = [order[i] for i, x in enumerate(std) if x > std_threshold or i < num_threshold]
  order.sort()
  return data[:, order]


def _needs_hessian(manifold):
  if hasattr(manifold, 'dissimilarity') and manifold.dissimilarity == 'precomputed':
    return True
  if hasattr(manifold, 'metric') and manifold.metric == 'precomputed':
    return True
  return False


@ut.images_to_uint8
@ut.timeit
def visualize_encoding(encodings, folder=None, meta={}, original=None, reconstruction=None):
  if np.max(original) < 10:
    original = (original * 255).astype(np.uint8)
  # print('np', np.max(original), np.max(reconstruction), np.min(original), np.min(reconstruction),
  #       original.dtype, reconstruction.dtype)
  file_path = None
  if folder:
    meta['postfix'] = 'pca'
    file_path = ut.to_file_name(meta, folder, 'jpg')
  encodings = manual_pca(encodings)

  if original is not None:
    assert len(original) == len(reconstruction)
    fig = plt.figure()

    # print('reco max:', np.max(reconstruction))
    column_picture, height = _stitch_images(original, reconstruction)
    subplot, proportion = (122, 1) if encodings.shape[1] <= 3 else (155, 3)
    picture = _reshape_column_image(column_picture, height, proportion=proportion)
    if picture.shape[-1] == 1:
      picture = picture.squeeze()
    plt.subplot(subplot).imshow(picture)

    visualize_encodings(encodings, file_name=file_path, fig=fig, grid=(3, 5), skip_every=5)
  else:
    visualize_encodings(encodings, file_name=file_path)


# @ut.timeit
def plot_encoding_crosssection(encodings, file_path, original=None, reconstruction=None, interactive=False):
  encodings = manual_pca(encodings)

  fig = _get_figure()
  if original is not None:
    assert len(original) == len(reconstruction),  (len(original), len(reconstruction))
    subplot, proportion = visualize_cross_section_with_reco(encodings)
    picture = stitch_side_by_side(original, reconstruction, proportion)
    subplot.imshow(picture)
  else:
    visualize_cross_section(encodings)
  if not interactive:
    save_fig(file_path, fig)


def plot_reconstruction(original, reconstruction, meta={'debug': 'true'}, interactive=False):
  # if not interactive:
  #   _get_figure()
  picture = stitch_side_by_side(original, reconstruction)
  plt.imshow(picture)
  if not interactive:
    file_path = ut.to_file_name(meta, FLAGS.save_path, 'jpg')
    save_fig(file_path)
  else:
    plt.draw()
    plt.pause(0.001)


# Image stitching


@ut.images_to_uint8
def stitch_side_by_side(original, reconstruction, proportion=1):
  """
  Stitch 2 lists of images together for convenient display in a single
  rectangular shape of given side proportion
  """
  assert np.max(original) >= 10, "some strange input to of the original pictures"
  column_picture, height = _stitch_images(original, reconstruction)
  picture = _reshape_column_image(column_picture, height, proportion=proportion)
  if picture.shape[-1] == 1:
    picture = picture.squeeze()
  return picture


def _stitch_images(*args):
  """Recieves one or many arrays of pictures and stitches them alongside into a column picture"""
  assert len(args) == 2
  lines, height, width, channels = args[0].shape
  min = 0  # int(args[0].mean())

  stack = args[0]
  if len(args) > 1:
    for i in range(len(args) - 1):
      stack = np.concatenate((stack, args[i + 1]), axis=2)
  # stack - array of lines of pictures (arr_0[0], arr_1[0], ...)
  # concatenate lines in one picture (height = tile_h * #lines)
  picture_lines = stack.reshape(lines * height, stack.shape[2], channels)
  picture_lines = np.hstack((
    picture_lines,
    np.ones((lines * height, 2, channels), dtype=np.uint8) * min))  # pad 2 pixels

  # slice/reshape to have better image proportions
  return picture_lines, height


def _reshape_column_image(column_picture, height, proportion=1):
  """
  Take an "column" image of independent horizontal segments of 'height'
  and reshape it to fit into given proportion

  |img1|
  |img2|    |img1 img4|
  |img3| => |img2 img5|
  |img4|    |img3     |
  |img5|

  """
  lines = int(column_picture.shape[0] / height)
  width = column_picture.shape[1]

  column_size = int(np.ceil(np.sqrt(lines * proportion * (width / height))))
  count = int(column_picture.shape[0] / height)
  _, _, channels = column_picture.shape

  picture = column_picture[0:column_size * height, :, :]

  for i in range(int(lines / column_size)):
    start, stop = column_size * height * (i + 1), column_size * height * (i + 2)
    if start >= len(column_picture):
      break
    if stop < len(column_picture):
      picture = np.hstack((picture, column_picture[start:stop]))
    else:
      last_column = np.vstack((
        column_picture[start:],
        np.ones((stop - len(column_picture), column_picture.shape[1], column_picture.shape[2]),
                dtype=np.uint8)))
      picture = np.hstack((picture, last_column))

  return picture


def _get_figure(fig=None):
  if fig is not None:
    return fig
  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 2)
  return fig


import matplotlib.ticker as ticker


def _plot_single_cross_section(data, select, subplot):
  data = data[:, select]
  # subplot.scatter(data[:, 0], data[:, 1], s=20, lw=0, edgecolors='none', alpha=1.0,
  subplot.plot(data[:, 0], data[:, 1], color='black', lw=1, alpha=0.4)
  subplot.plot(data[[-1, 0], 0], data[[-1, 0], 1], lw=1, alpha=0.8, color='red')
  subplot.scatter(data[:, 0], data[:, 1], s=4, alpha=1.0, lw=0.5,
                  c=_build_radial_colors(len(data)),
                  marker=".",
                  cmap=plt.cm.Spectral)
  # data = np.vstack((data, np.asarray([data[0, :]])))
  # subplot.plot(data[:, 0], data[:, 1], alpha=0.4)

  subplot.set_xlabel('feature %d' % select[0])
  subplot.set_ylabel('feature %d' % select[1])
  subplot.set_xlim([-0.05, 1.05])
  subplot.set_ylim([-0.05, 1.05])
  subplot.xaxis.set_ticks([0, 1])
  subplot.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
  subplot.yaxis.set_ticks([0, 1])
  subplot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))

def _plot_single_cross_section_3d(data, select, subplot):
  data = data[:, select]
  # subplot.scatter(data[:, 0], data[:, 1], s=20, lw=0, edgecolors='none', alpha=1.0,
  subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=1, alpha=0.4)
  subplot.plot(data[[-1, 0], 0], data[[-1, 0], 1], data[[-1, 0], 2], lw=1, alpha=0.8, color='red')
  subplot.scatter(data[:, 0], data[:, 1], data[:, 2], s=4, alpha=1.0, lw=0.5,
                  c=_build_radial_colors(len(data)),
                  marker=".",
                  cmap=plt.cm.Spectral)
  data = data[0::10]
  # subplot.plot(data[:, 0], data[:, 1], data[:, 2], color='black', lw=2, alpha=0.8)

  # data = np.vstack((data, np.asarray([data[0, :]])))
  # subplot.plot(data[:, 0], data[:, 1], alpha=0.4)

  subplot.set_xlabel('feature %d' % select[0])
  subplot.set_ylabel('feature %d' % select[1])
  subplot.set_xlim([-0.01, 1.01])
  subplot.set_ylim([-0.01, 1.01])
  subplot.set_zlim([-0.01, 1.01])
  subplot.xaxis.set_ticks([0, 1])
  subplot.yaxis.set_ticks([0, 1])
  subplot.zaxis.set_ticks([0, 1])
  subplot.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
  subplot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))


@ut.deprecated
def visualize_cross_section(embeddings):
  features = embeddings.shape[-1]
  size = features - 1
  for i in range(features):
    for j in range(i + 1, features):
      pos = i * size + j
      subplot = plt.subplot(size, size, pos)
      _plot_single_cross_section(embeddings, [i, j], subplot)

  if features >= 3:
    pos = (size + 1) * size - size + 1
    subplot = plt.subplot(size + 1, size, pos)
    _plot_single_cross_section(embeddings, [0, 1], subplot)
  return size


def visualize_cross_section_with_reco(embeddings):
  features = embeddings.shape[-1]
  size = features - 1
  for i in range(features):
    for j in range(i + 1, features):
      pos = i * (size + 1) + j
      subplot = plt.subplot(size, size + 1, pos)
      _plot_single_cross_section(embeddings, [i, j], subplot)
  reco_subplot = plt.subplot(1, size + 1, size + 1)

  if size >= 2:
    single_size = size if size < 4 else int(size/2)
    pos = single_size * (single_size + 1) - (single_size + 1) + 1
    subplot = plt.subplot(single_size, single_size+1, pos, projection='3d')
    _plot_single_cross_section_3d(embeddings, [0, 1, 2], subplot)
  return reco_subplot, size


# cross section end


def visualize_encodings(encodings, file_name=None,
                        grid=None, skip_every=999, fast=False, fig=None, interactive=False):
  encodings = manual_pca(encodings)
  if encodings.shape[1] <= 3:
    return print_data_only(encodings, file_name, fig=fig, interactive=interactive)

  encodings = encodings[0:720]
  hessian_euc = dist.squareform(dist.pdist(encodings[0:720], 'euclidean'))
  hessian_cos = dist.squareform(dist.pdist(encodings[0:720], 'cosine'))
  grid = (3, 4) if grid is None else grid
  project_ops = []

  n = 2
  project_ops.append(("LLE ltsa       N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='ltsa')))
  project_ops.append(("LLE modified   N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='modified')))
  project_ops.append(('MDS euclidean  N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))
  project_ops.append(("TSNE 30/2000   N:%d" % n, TSNE(perplexity=30, n_components=n, init='pca', n_iter=2000)))
  n = 3
  project_ops.append(("LLE ltsa       N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='ltsa')))
  project_ops.append(("LLE modified   N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='modified')))
  project_ops.append(('MDS euclidean  N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))
  project_ops.append(('MDS cosine     N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))

  plot_places = []
  for i in range(12):
    u, v = int(i / (skip_every - 1)), i % (skip_every - 1)
    j = v + u * skip_every + 1
    plot_places.append(j)

  fig = fig if fig is not None else plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * grid[0] / 1.,
                      fig.get_size_inches()[1] * grid[1] / 2.0)

  for i, (name, manifold) in enumerate(project_ops):
    is3d = 'N:3' in name

    try:
      if is3d:
        subplot = plt.subplot(grid[0], grid[1], plot_places[i], projection='3d')
      else:
        subplot = plt.subplot(grid[0], grid[1], plot_places[i])

      data_source = encodings if not _needs_hessian(manifold) else \
        (hessian_cos if 'cosine' in name else hessian_euc)
      projections = manifold.fit_transform(data_source)
      scatter(subplot, projections, is3d, _build_radial_colors(len(data_source)))
      subplot.set_title(name)
    except:
      print(name, "Unexpected error: ", sys.exc_info()[0], sys.exc_info()[1] if len(sys.exc_info()) > 1 else '')

  visualize_data_same(encodings, grid=grid, places=plot_places[-4:])
  if not interactive:
    save_fig(file_name, fig)
  ut.print_time('visualization finished')


def save_fig(file_name, fig=None):
  if not file_name:
    plt.show()
  else:
    plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None)
    # plt.close('all')


def _random_split(sequence, length, original):
  if sequence is None or len(sequence) < length:
    sequence = original.copy()
  sequence = np.random.permutation(sequence)
  return sequence[:length], sequence[length:]


@ut.deprecated
def visualize_data_same(data, grid, places):
  assert len(places) == 4

  all_dimensions = np.arange(0, data.shape[1]).astype(np.int8)
  first_proj, left = _random_split(None, 2, all_dimensions)
  first_color_indexes, _ = _random_split(left, 3, all_dimensions)
  first_color = _data_to_colors(data, first_color_indexes - 2)

  second_proj, left = _random_split(left, 2, all_dimensions)
  second_color = _build_radial_colors(len(data))

  third_proj, left = _random_split(left, 3, all_dimensions)
  third_color_indexes, _ = _random_split(left, 3, all_dimensions)
  third_color = _data_to_colors(data, third_color_indexes)

  forth_proj = np.argsort(data.std(axis=0))[::-1][0:3]
  forth_color = _build_radial_colors(len(data))

  for i, (projection, color) in enumerate([
    (first_proj, first_color),
    (second_proj, second_color),
    (third_proj, third_color),
    (forth_proj, forth_color)]
  ):
    points = np.transpose(data[:, projection])

    if len(projection) == 2:
      subplot = plt.subplot(grid[0], grid[1], places[i])
      subplot.scatter(points[0], points[1], c=color, cmap=COLOR_MAP, picker=PICKER_SENSITIVITY)
    else:
      subplot = plt.subplot(grid[0], grid[1], places[i], projection='3d')
      subplot.scatter(points[0], points[1], points[2], c=color, cmap=COLOR_MAP, picker=PICKER_SENSITIVITY)
    subplot.set_title('Data %s %s' % (str(projection), 'sequntial color' if i % 2 == 1 else ''))


@ut.deprecated
def visualize_data_same_deprecated(data, grid, places, dims_as_colors=False):
  assert len(places) == 4
  dimensions = np.arange(0, np.min([6, data.shape[1]])).astype(np.int)
  assert len(dimensions) == data.shape[1] or len(dimensions) == 6
  projections = [dimensions[x] for x in [[0, 1], [-1, -2], [0, 1, 2], [-1, -2, -3]]]
  colors = _build_radial_colors(len(data))

  for i, dims in enumerate(projections):
    points = np.transpose(data[:, dims])
    if dims_as_colors:
      colors = _data_to_colors(np.delete(data.copy(), dims, axis=1))

    if len(dims) == 2:
      subplot = plt.subplot(grid[0], grid[1], places[i])
      subplot.scatter(points[0], points[1], c=colors, cmap=COLOR_MAP)
    else:
      subplot = plt.subplot(grid[0], grid[1], places[i], projection='3d')
      subplot.scatter(points[0], points[1], points[2], c=colors, cmap=COLOR_MAP)
    subplot.set_title('Data %s' % str(dims))


def _duplicate_array(array, repeats=None, total_length=None):
  assert repeats is not None or total_length is not None

  if repeats is None:
    repeats = int(np.ceil(total_length / len(array)))
  res = array.copy()
  for i in range(repeats - 1):
    res = np.concatenate((res, array))
  return res if total_length is None else res[:total_length]


def _build_radial_colors(length):
  colors = np.arange(0, 180)
  colors = np.concatenate((colors, colors[::-1]))
  colors = _duplicate_array(colors, total_length=length)
  return colors


def _data_to_colors(data, indexes=None):
  color_data = data[:, indexes] if indexes is not None else data
  shape = color_data.shape

  if shape[1] < 3:
    add = 3 - shape[1]
    add = np.ones((shape[0], add)) * 0.5
    color_data = np.concatenate((color_data, add), axis=1)
  elif shape[1] > 3:
    color_data = color_data[:, 0:3]

  if np.max(color_data) <= 1:
    color_data *= 256
  color_data = color_data.astype(np.int32)
  assert np.mean(color_data) <= 256
  color_data[color_data > 255] = 255
  color_data *= np.asarray([256 ** 2, 256, 1])

  color_data = np.sum(color_data, axis=1)
  color_data = ["#%06x" % c for c in color_data]
  return color_data


def visualize_available_data(root='./', reembed=True, with_mds=False):
  tf.app.flags.DEFINE_string('suffix', 'grid', '')
  FLAGS.suffix = '__' if with_mds else 'grid'
  assert os.path.exists(root)

  files = _list_embedding_files(root, reembed=reembed)
  total_files = len(files)

  for i, file in enumerate(files):
    print('%d/%d %s' % (i + 1, total_files, file))
    data = np.loadtxt(file)
    png_name = file.replace('.txt', '_pca.png')

    visualize_encodings(data, file_name=png_name)


def _list_embedding_files(root, reembed=False):
  ecndoding_files = []
  for root, dirs, files in os.walk(root):
    if '/tmp' in root:
      for file in files:
        if '.txt' in file and 'meta' not in file:
          full_path = os.path.join(root, file)
          if not reembed:
            vis_file = full_path.replace('.txt', '_pca.png')
            if os.path.exists(vis_file):
              continue
          ecndoding_files.append(full_path)
  return ecndoding_files


if __name__ == '__main__':
  path = '../../encodings__e|500__z_ac|96.5751.txt'
  x = np.loadtxt(path)
  # x = np.random.rand(100, 5)
  x = manual_pca(x)
  x = x[:360]
  visualize_cross_section_with_reco(x)
  plt.tight_layout()
  plt.show()
