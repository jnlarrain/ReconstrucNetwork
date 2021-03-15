import tensorflow as tf
import numpy as np

g1 = tf.Graph()
g2 = tf.Graph()
g3 = tf.Graph()

# create a, b in graph g1
with g3.as_default():
    with g1.as_default():
        a = tf.Variable(np.array([1, 2], dtype=np.float32))
        b = tf.Variable(np.array([2, 2], dtype=np.float32))
        result1 = a + b
    # create c, d in graph g2
    with g2.as_default():
        c = tf.Variable(np.array([1, 2], dtype=np.float32))
        d = tf.Variable(np.array([2, 2], dtype=np.float32))
        result2 = c + d
    result3 = result1+result2
# create session
with tf.Session(graph=g1) as sess:
    out = sess.run(result1)
    print(out)
with tf.Session(graph=g2) as sess:
    out = sess.run(result2)
    print(out)

with tf.Session(graph=g3) as sess:
    out = sess.run(result3)
    print(out)

