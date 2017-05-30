'''
LinearModel using tensorflow
Data Credit: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np

class LinearModel:
	#Initialize model parameters:
									
	def initialize_model_params(self,n_samples , epoch, learning_rate, mini_batch_size):
		self.n_samples = n_samples
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.mini_batch_size = mini_batch_size
		
		#1 Layer Neural network
		self.hidden_layer1 = { 'weights': tf.Variable(tf.random_normal([1, 1])),
				 'bias' : tf.Variable(tf.random_normal(shape = [self.n_samples, 1]))
				}

		#Input and output placeholder for tensorflow 
		self.x = tf.placeholder(tf.float32, shape=[None, 1])
		self.y = tf.placeholder(tf.float32, shape=[None, 1])
		
	#Feed forward
	def feed_forward(self, input_data):
		
		output_layer = tf.add(tf.matmul(input_data, self.hidden_layer1['weights']) , self.hidden_layer1['bias'])
		return output_layer
		
	def train_network(self, train_X, train_Y):
		predicted_val = self.feed_forward(self.x)
		cost = tf.reduce_mean(tf.pow(predicted_val-self.y, 2)/self.n_samples)
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for epoch in range(self.epoch):
				epoch_loss = 0
				for i in range(self.mini_batch_size):
					_, c = sess.run([optimizer, cost], feed_dict= {self.x:train_X, self.y:train_Y})
					epoch_loss += c
					print 'epoch loss' 
			
						
	#Test the model:
	def test_model(self, test_X, test_Y):
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			predicted_val = self.feed_forward(self.x)
			correct_prediction = tf.equal(tf.argmax(predicted_val,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
			print('acc', accuracy.eval({self.x: test_X, self.y: test_Y}))		
		 # Graphic display
		
		

if __name__ == '__main__':
	
	# Training Data for model
	train_X = np.matrix([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1], dtype = np.float32)
	train_X = np.transpose(train_X)
	train_Y = np.matrix([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3], dtype = np.float32)
	train_Y = np.transpose(train_Y)
	
	
	#Model parameters
	n_samples = 17
	batch_size = 100
	epochs = 1
	learning_rate = 0.001
	mini_batch_size = 1
	
	
	lin_model_obj = LinearModel()
	
	#Initalize model parameters
	print 'Initializing parameters'
	lin_model_obj.initialize_model_params(n_samples, epochs, learning_rate, mini_batch_size)
	print 'Network params initialized'
	
	#Train the network
	print 'Training the model'
	lin_model_obj.train_network(train_X, train_Y)
	
	#Test data for the model
	test_X = np.matrix([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1, 6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1, 1.1], dtype = np.float32)
	test_X = np.transpose(test_X)
	test_Y = np.matrix([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03,6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1, 1.1], dtype = np.float32)
	test_Y = np.transpose(test_Y)
	
	print 'Testing the model'
	lin_model_obj.test_model(test_X, test_Y)
	
	#Done 
	print 'Done'
	