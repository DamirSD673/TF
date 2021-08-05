import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class DENSE(tf.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((self.output, x.shape[0]), stddev=0.2, name='w')
            self.b = tf.zeros([self.output], dtype=tf.float32, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = self.w @ x + self.b
        return y


a = tf.constant([1.0, 2.0], shape=(2, 1))
print(a)
b = tf.constant([[1.0, 2.0]])
print('b = ', b)
print(a.shape[0])

Neuron = DENSE(1)
#print(Neuron(a))

x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2)) # training data
y_train = [a + b for a, b in x_train]
print('x_train', x_train)
print('y_train = ', y_train)
print('x , y = ', list(zip(x_train, y_train)))

loss = lambda x, y: tf.reduce_mean(tf.square(x - y)) # Create anonymous function
opt = tf.optimizers.Adam(learning_rate=0.01)

ITERATIONS = 3
for i in range(ITERATIONS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=1)
        y = tf.constant(y, shape=(1, 1))

        with tf.GradientTape() as tape:
            e = loss(y, Neuron(x))

        grads = tape.gradient(e, Neuron.trainable_variables)
        opt.apply_gradients(zip(grads, Neuron.trainable_variables))

    #print(e.numpy())

print(Neuron(a))