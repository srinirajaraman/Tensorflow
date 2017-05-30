'''
Adding 2 matrices using Tensorflow
'''
import tensorflow as tf
import numpy as np
matrix_one = tf.constant(np.arange(0, 9, dtype = np.int32), shape = [3, 3])
matrix_two =  tf.constant(np.arange(0, 9, dtype = np.int32), shape = [3, 3])
result = tf.matmul(matrix_one, matrix_two)
with tf.Session() as sess:
	print(sess.run(result))