'''
Hello world from tensorflow
'''
import tensorflow as tf
hello_world = tf.constant('Hello World Tensorflow!')
with tf.Session() as sess:
	print(sess.run(hello_world))