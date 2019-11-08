# import numpy as np
# a = np.array([[1, 2], [3, 4]])
# b = a[0]
# c = a[1]
# print(b)
# print(c)
# print(c.shape)
# class hparam:
#     def __int__(self):
#         self.d = 5
#
# hp = hparam()
# hp.d = 4
# print(hp.d)
#
# a = 'haha 123'
# print(a.find('q'))
# print(a.find('ah'))
import tensorflow as tf
import numpy as np
a = tf.placeholder(shape=[5, 12, 7], dtype=tf.float32)
b = tf.layers.conv1d(a, filters=10, kernel_size=6, padding='VALID', data_format='channels_last')
c = tf.layers.max_pooling1d(a, pool_size=5, strides=1, padding='VALID')
print(b.get_shape().as_list())
print(c.get_shape().as_list())
with tf.Session() as sess:
    _a = np.random.random(size=[5, 12, 7])
    sess.run(tf.global_variables_initializer())
    _b, _c = sess.run((b, c), feed_dict={a: _a})