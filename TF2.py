import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w = tf.Variable(tf.random.uniform((3, 2), 1, 4, dtype=tf.int32))
b = tf.Variable(tf.random.uniform((2, 1), 1, 3, dtype=tf.int32))
w = tf.Variable(tf.cast(w, dtype=tf.float32))
b = tf.Variable(tf.cast(b, dtype=tf.float32))
x = tf.Variable([[-2.0, 1.0, 3.0]])

with tf.GradientTape() as tape:
    y = x @ w + tf.transpose(b)
    loss = tf.reduce_mean(y ** 2)

df = tape.gradient(loss, [w, b])
print('Gradient 1 w = ', df[0], '\n', 'Gradient 2 w =', df[1])
