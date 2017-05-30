'''
This code tests a 1 layer neural network with standard back propagation based on Tensorflow 
Tensorflow credits: https://www.tensorflow.org/ 
Credit: https://pythonprogramming.net/
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class NeuralNetwork:
	#Initialize model parameters:
	def initialize_model_params(self, ip_data, n_input_nodes, n_nodes_h1, n_classes, batch_size, epoch, learning_rate, mini_batch_size):
		self.ip_data = ip_data
		self.n_input_nodes = n_input_nodes
		self.n_nodes_h1 = n_nodes_h1
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.mini_batch_size = mini_batch_size
		
		#1 Layer Neural network
		self.hidden_layer1 = { 'weights': tf.Variable(tf.random_normal([self.n_input_nodes, n_nodes_h1])),
							'bias' : tf.Variable(tf.random_normal([self.n_nodes_h1]))
						}
					
		self.output_layer =  { 'weights': tf.Variable(tf.random_normal([self.n_nodes_h1,self.n_classes])),
							'bias' : tf.Variable(tf.random_normal([self.n_classes]))
						}
		
		#Input and output placeholder for tensorflow 
		self.x = tf.placeholder('float', [None, self.n_input_nodes])
		self.y = tf.placeholder('float', [None, self.n_classes])

	#Feed forward
	def feed_forward(self, input_data):

		layer_1 = tf.add(tf.matmul(input_data, self.hidden_layer1['weights']) , self.hidden_layer1['bias'])
		layer_1 = tf.nn.sigmoid(layer_1)                                                           ` 
		output_layer = tf.matmul(layer_1, self.output_layer['weights']) + self.output_layer['bias']
		
		return output_layer
	
#Train the network
	def train_network(self):
	
		predicted_val = self.feed_forward(self.x)
		cost = tf.nn.softmax_cross_entropy_with_logits(predicted_val, self.y)
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
		
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for epoch in range(self.epoch):
				epoch_loss = 0
				for i in range(self.mini_batch_size):
					epoch_x, epoch_y = self.ip_data.train.next_batch(batch_size)
					_, c = sess.run([optimizer, cost], feed_dict= {self.x:epoch_x, self.y:epoch_y})
					epoch_loss += c
				print 'epoch', epoch_loss
						

	def test_model(self):
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			predicted_val = self.feed_forward(self.x)
			correct_prediction = tf.equal(tf.argmax(predicted_val,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
			print('acc', accuracy.eval({self.x: self.ip_data.test.images, self.y: self.ip_data.test.labels}))		
				

if __name__ == '__main__':
	#Model parameters
	n_input_nodes = 784; 
	n_nodes_h1 = 100
	n_classes = 10
	batch_size = 100
	epochs = 100
	learning_rate = 0.001
	mini_batch_size = 100
	
	mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
	
	obj_bp = NeuralNetwork()
	
	print 'Initializing parameters'
	#Initalize model parameters
	obj_bp.initialize_model_params(mnist, n_input_nodes , n_nodes_h1, n_classes, batch_size, epochs, learning_rate, mini_batch_size)
	print 'Network params initialized'
	
	print 'Training the model'
	#Train the network
	obj_bp.train_network()
	
	print 'Testing the model'
	#Test the network
	obj_bp.test_model()
	print 'Done'
	