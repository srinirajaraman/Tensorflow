'''
Recurrent neural network with standard back propagation based on Tensorflow 
Credits: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class RecurrentNeuralNetwork:
	#Initialize model parameters:
	def initialize_model_params(self, n_input_nodes , n_hidden_nodes, n_output_nodes, n_steps, learning_rate, epoch, batch_size):
		self.n_input_nodes = n_input_nodes
		self.n_hidden_nodes = n_hidden_nodes
		self.n_steps = n_steps
		self.n_classes = n_output_nodes
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.batch_size = batch_size
		with tf.variable_scope("lstm_cell_weights"):
	 		self.bias = tf.get_variable("bias", shape = (self.n_classes), initializer=tf.random_normal_initializer())
	 	with tf.variable_scope("lstm_cell_bias"):
	 		self.weights = tf.get_variable("weights", shape = (self.n_hidden_nodes, self.n_classes), initializer=tf.random_normal_initializer())
		self.x = tf.placeholder("float", [None, self.n_steps, self.n_input_nodes])
		self.y = tf.placeholder("float", [None, self.n_classes])

	#Feed forward
	def feed_forward(self, input_data):
		with tf.variable_scope("lstm_cell") as scope:
			# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
			x = tf.unstack(input_data, n_steps, 1)
			# Define a lstm cell with tensorflow
			lstm_cell = rnn.BasicLSTMCell(self.n_hidden_nodes, forget_bias=1.0)
			# Get lstm cell output
			outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

			# Linear activation, using rnn inner loop last output
			output_layer = tf.matmul(outputs[-1], self.weights) + self.bias

		return output_layer
	#Train the network
	def train_network(self, train_X, test_data):

		predicted_val = self.feed_forward(self.x)
		cost = tf.nn.softmax_cross_entropy_with_logits(logits = predicted_val, labels = self.y)
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(self.epoch):
				epoch_loss = 0
				epoch_x, epoch_y = train_X.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, self.n_steps, self.n_input_nodes))
				val, c = sess.run([optimizer, cost], feed_dict= {self.x:epoch_x, self.y:epoch_y})
				epoch_loss += c
				if(epoch % 2000 == 0):
					print 'epoch', epoch_loss
			test_len = 128
			test_data_images = test_data.images[:test_len].reshape((-1,self.n_steps, self.n_input_nodes))
			test_label = test_data.labels[:test_len]
			#predicted_val = self.feed_forward(self.x)
			correct_prediction = tf.equal(tf.argmax(predicted_val,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
			print('acc', accuracy.eval({self.x: test_data_images, self.y: test_label}))

			
	
if __name__ == '__main__':
	n_input_nodes = 28
	n_hidden_nodes = 28
	n_classes = 10
	n_steps = 28
	learning_rate = 0.001
	batch_size = 128
	epoch = 10000

	obj_rnn = RecurrentNeuralNetwork()

	#Initalize model parameters
	print 'Initializing parameters'
	obj_rnn.initialize_model_params(n_input_nodes , n_hidden_nodes, n_classes, n_steps, learning_rate, epoch, batch_size)
	print 'Network params initialized'

	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	#Train the network
	print 'Training the model'
	obj_rnn.train_network(mnist.train, mnist.test)

	#Test the network
	print 'Done'
	