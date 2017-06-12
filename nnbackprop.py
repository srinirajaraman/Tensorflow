'''
This code tests a 1 layer neural network with standard back propagation based on Tensorflow 
Tensorflow credits: https://www.tensorflow.org/ 
Credit: https://pythonprogramming.net/
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class NeuralNetwork:
	#Initialize model parameters:
	def initialize_model_params(self, n_input_nodes, n_nodes_h1, n_classes, batch_size, epoch, learning_rate):
		self.n_input_nodes = n_input_nodes
		self.n_nodes_h1 = n_nodes_h1
		self.n_classes = n_classes
		self.batch_size = batch_size
		self.epoch = epoch
		self.learning_rate = learning_rate
		
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
		layer_1 = tf.nn.sigmoid(layer_1)
		output_layer = tf.matmul(layer_1, self.output_layer['weights']) + self.output_layer['bias']
		
		return output_layer
	
#Train the network
	def train_network(self, train_data, test_data):
	
		predicted_val = self.feed_forward(self.x)
		cost = tf.nn.softmax_cross_entropy_with_logits(predicted_val, self.y)
		optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
		#Training the model
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			for epoch in range(self.epoch):
				epoch_loss = 0
				epoch_x, epoch_y = train_data.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict= {self.x:epoch_x, self.y:epoch_y})
				epoch_loss += c
				if(epoch % 100 == 0):
					print 'epoch', epoch_loss
			
			#Testing the model
			print('Testing the model')
			predicted_val = self.feed_forward(self.x)
			correct_prediction = tf.equal(tf.argmax(predicted_val,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
			print('acc', accuracy.eval({self.x: test_data.images, self.y: test_data.labels})))			
			

if __name__ == '__main__':
	#Model parameters
	n_input_nodes = 784; 
	n_nodes_h1 = 100
	n_classes = 10
	batch_size = 100
	epochs = 100
	learning_rate = 0.001
	
	mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
	
	obj_bp = NeuralNetwork()
	
	print 'Initializing parameters'
	#Initalize model parameters
	obj_bp.initialize_model_params(n_input_nodes , n_nodes_h1, n_classes, batch_size, epochs, learning_rate)
	print 'Network params initialized'
	
	print 'Training the model'
	#Train the network
	obj_bp.train_network(mnist.train, mnist.test)
	print 'Done'
	