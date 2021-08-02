import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import time

N_POINTS = 1000

x = tf.random.uniform(shape=[N_POINTS], minval=0, maxval=10)
n = tf.random.normal(shape = [N_POINTS], mean=0, stddev=0.2)
k_real = 2.0
b_real = 1.3

y = (k_real * x + b_real) + n

plt.scatter(x, y, s=2)
plt.show()
k = tf.Variable(0.0)
b = tf.Variable(0.0)

tf.print(b)

ITERATIONS = 100
lyambda = 0.02
plt.ion()
fig, ax1 = plt.subplots(nrows=1, ncols=1)
ax1.grid(True)
#ax2.grid(True)

point1 = ax1.scatter(0, 0, c='r')
#point2 = ax2.scatter(0, 0, c='r')
for i in range(ITERATIONS):
    with tf.GradientTape() as tape:
        y_hat = k * x + b
        loss = tf.reduce_mean(tf.square(y - y_hat))

    dk, db = tape.gradient(loss, [k, b])
    k.assign_sub(lyambda * dk)
    b.assign_sub(lyambda * db)
    point1.set_offsets([k, loss])
    #point2.set_offsets([b, loss])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
ax1.scatter(k, loss, c='r')
#ax2.scatter(b, loss, c='r')
plt.show()

y_new = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_new, c='r', s=2)
print('k_hat = ', k, '\n', 'b_hat = ', b)
plt.show()
