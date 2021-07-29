import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

Z = tf.zeros((3, 3))

print(Z)

Z2 = tf.zeros_like(Z)

eye = tf.eye(3, (2))

r = tf.random.uniform((4, 3), 1, 8, dtype=tf.int32)
r2 = tf.random.uniform((4, 3), 1, 4, dtype=tf.int32)
r_d = tf.divide(r, r2)
M = tf.reduce_mean(r_d, axis=0)
print(M)