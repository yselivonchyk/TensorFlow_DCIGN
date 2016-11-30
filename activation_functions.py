import tensorflow as tf

sigmoid =   type('Fake', (object,), { "func": tf.nn.sigmoid, "min": 0,  'max': 1})
tanh =      type('Fake', (object,), { "func": tf.nn.tanh,    "min": -1, 'max': 1})
relu =      type('Fake', (object,), { "func": tf.nn.relu,    "min": 0, 'max': 1})
