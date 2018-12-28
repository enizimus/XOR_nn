import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.random as rd
import time

save_pth = './models/xor_model.ckpt'

n_hidd = 2
n_in = 2
learning_rate = 0.1

# training data and labels
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])


w1 = tf.Variable(tf.random_uniform([n_in, n_hidd], -1, 1), trainable=True)
b1 = tf.Variable(tf.zeros(n_hidd))
w2 = tf.Variable(tf.random_uniform([n_hidd, 1], -1, 1), trainable=True)
b2 = tf.Variable(tf.zeros(1))

x = tf.placeholder(shape=(None, X.shape[1]), dtype=tf.float32)
z = tf.matmul(x, w1) + b1
a = tf.nn.relu(z)

y = tf.nn.sigmoid(tf.matmul(a, w2) + b2)
z_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)

cost  = tf.reduce_mean(((z_ * tf.log(y)) + ((1 - z_) * tf.log(1.0 - y))) * -1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(z_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction))

init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

for i in range(5000):
    sess.run(train_step, feed_dict={x: X, z_: Y})

    loss = sess.run(cost, feed_dict={x: X, z_: Y})
    #acc = sess.run(accuracy, feed_dict={x: X, z_: Y})

    if i%1000 == 0:
        print('Iteration : {}, loss : {:.3f}'.format(i, loss))

saver.save(sess, save_pth)

# test data and test labels
X_test = np.array([[1., 1.], [0., 0.], [1., 0.], [1., 1.], [1., 0], [0., 1.], [0., 0.], [0., 1.], [0., 1.], [1., 1.], [1., 0.]])
Y_test = np.array([[0.], [0.], [1.], [0.], [1.], [1.] , [0.], [1.], [1.], [0.], [1.]], dtype=np.float32)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(z_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc = sess.run(accuracy, feed_dict={x: X_test, z_: Y_test})
print('Acc = {:.3f}%'.format(acc*100))

