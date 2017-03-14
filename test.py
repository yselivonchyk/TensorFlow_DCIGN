import numpy as np
import tensorflow as tf
import visualization as vis


# def minibatch_test():
#   x = np.arange(0, 100, 1)
#
#   inp = model.input.Input(x)
#   for j in range(100):
#     print(inp.generate_minibatch(17))

#
# def batcher_test():
#   x = np.arange(0, 100, 1)
#   y = np.arange(200, 300, 1)
#   bs = 7
#   batcher = Batcher.Batcher(bs, x, y)
#   sum = 0
#   for j in range(100):
#     u, v = batcher.get_batch()
#     sum += np.sum(u) + np.sum(v)
#     assert np.sum(v-u) == 200*bs
#   assert sum/bs == np.sum(x) + np.sum(y)
#
# def to_file_name_test():
#   print(ut.to_file_name({
#     'suf': 'test',
#     'e': 20,
#     'act': tf.nn.sigmoid,
#     'opt': tf.train.GradientDescentOptimizer(learning_rate=0.001),
#     'lr': 0.001,
#     'init': tf.truncated_normal_initializer(stddev=0.35)
#   }))
#   print(ut.to_file_name({
#     'suf': 'test',
#     'e': 20,
#     'act': tf.nn.sigmoid,
#     'opt': tf.train.GradientDescentOptimizer(learning_rate=0.001),
#     'lr': 0.000001,
#     'init': tf.truncated_normal_initializer
#   }))
#
#
# def test_time():
#   ut.print_time('one')
#   ut.print_time('two')
#
#
# def test_activation():
#   print(act.sigmoid, act.sigmoid.func, act.sigmoid.max, act.sigmoid.min)
#
#
# def test_ds_scale():
#   ds = [-4.0, -2.0, 2.0, 4.0]
#   scaled = ut.rescale_ds(ds, -2, -1)
#   assert (scaled[0] - scaled[1]) * 2 == (scaled[1] - scaled[2])
#   assert min(scaled) == -2
#   assert max(scaled) == -1
#
#
# def test_manual_pca():
#   d = np.random.rand(100, 2)
#   d = np.hstack((np.zeros((100, 5)), d))
#   proj = vis.manual_pca(d)
#   assert proj.shape[1] == 3
#   assert proj[0, 0] != 0
#   assert proj[0, -1] == 0
#   std = np.std(proj, axis=0)
#   assert std[0] > std[1]


# to_file_name_test()
# test_manual_pca()
#
# test_activation()
# test_ds_scale()
# # test_time()
# to_file_name_test()
# batcher_test()

import network_utils as nut



def save(shape):
  try:
    global x
    inp = tf.Variable(x)
    op = tf.nn.max_pool_with_argmax(inp,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
    op1 = nut.unpool(op[0], op[1])
    op2 = nut.unpool(op[0], tf.Variable(fake_args(shape)))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      x, args = sess.run([op])[0]
      x2 = sess.run([op1])[0]
      x3 = sess.run([op2])[0]

      np.savetxt(open('./x.txt', 'wb'), x, fmt='%s')
      np.savetxt(open('./a.txt', 'wb'), args, fmt='%s')
      np.savetxt(open('./x2.txt', 'wb'), x2, fmt='%s')
      np.savetxt(open('./x2.txt', 'wb'), x3, fmt='%s')
      np.save('./a', args)
      np.save('./x', x)
      np.save('./x2', x2)
      np.save('./x3', x3)
  except:
    pass


def fake_arg_max(shape, stride=2):
  assert shape[1] % stride == 0 and shape[2] % stride == 0, 'Smart padding is not supported'
  mask = np.arange(np.prod(shape[1:]))
  mask = mask.reshape(shape[1:])
  mask = mask[::stride, ::stride, :]
  mask = np.tile(mask, (shape[0], 1, 1, 1))
  return mask


shape = [2, 6, 6, 2]
total = np.prod(shape)
x = np.arange(total, dtype=np.float32) + 1000
x = x
x = np.reshape(x, shape)

print(fake_arg_max(shape))

save(shape)

print(x.squeeze())

print('\nx0\n', x[:,:,:,0])
print('\nx1\n', x[:,:,:,1])

x = np.load('/mnt/code/vd/TensorFlow_DCIGN/x.npy')
a = np.load('/mnt/code/vd/TensorFlow_DCIGN/a.npy')
x2 = np.load('/mnt/code/vd/TensorFlow_DCIGN/x2.npy')
x3 = np.load('/mnt/code/vd/TensorFlow_DCIGN/x3.npy')

print(x.shape)
print(a.shape)

print('\nx0\n', x[:,:,:,0])
print('\nx1\n', x[:,:,:,1])

print('\n!x0\n', x2[:,:,:,0])
print('\n!x1\n', x2[:,:,:,1])

print('\n!77x0\n', x3[:,:,:,0])
print('\n!77x1\n', x3[:,:,:,1])

print('\na0\n', a[:,:,:,0])
print('\na1\n', a[:,:,:,1])
print('--!@1311\n', a[:,0,0,0])
# print(x.squeeze())
# print(a.squeeze())

a = fake_args(shape)


print('--a0\n', a[:,:,:,0])
print('--a1\n', a[:,:,:,1])
print('--a1\n', a[:,0,0,0])
# print('__')
# print(a.squeeze())