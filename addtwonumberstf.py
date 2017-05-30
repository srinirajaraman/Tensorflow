'''
Adding 2 numbers using Tensorflow
'''
import tensorflow as tf
number_one = tf.constant(7)
number_two = tf.constant(8)
result = number_one * number_two
with tf.Session() as sess:
	print(sess.run(result))