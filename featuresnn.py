#imporitng the required library
import tensorflow as tf
from features import create_labels
import numpy as np

#importing the input data.
trainx, trainy, testx, testy = create_labels('pos.txt', 'neg.txt')

#defining the hidden layers.
n_nodes_hl1 = 500
n_nodes_hl2 = 100
n_nodes_hl3 = 500

#defining the number of input classes and batch size.
n_classes = 2 
batch_size = 100

#Inserts a placeholder for a tensor that will be always fed.
#tf.placeholder(dtype, shape=None, name=None)
x = tf.placeholder('float', [None, len(trainx[0])])
y = tf.placeholder('float')

#Defining the neural network.
def neural_network_model(data):
	#defining the hidden layers.
	#the weights and biases are stored in tf.variables.
	h1_layer = {'weights':tf.Variable(tf.random_normal([len(trainx[0]), n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	h2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	h3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	op_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
			'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	l1 = tf.add(tf.matmul(data, h1_layer['weights']),  h1_layer['biases'])
	l1 = tf.nn.relu(l1) #using ReLu (rectified linear unit) activation function.
	
	l2 = tf.add(tf.matmul(l1, h2_layer['weights']), h2_layer['biases'])
	l2 = tf.nn.relu(l2) #using ReLu (rectified linear unit) activation function.

	l3 = tf.add(tf.matmul(l2, h3_layer['weights']), h3_layer['biases'])
	l3 = tf.nn.relu(l3) #using ReLu (rectified linear unit) activation function.

	op = tf.matmul(l3, op_layer['weights']) + op_layer['biases']
	#returning the model
	return op

#Training the neural network.
def trainNN (x):
	#this builds the graph as far as needed to return the tensor that would contain the output predictions.
	prediction = neural_network_model(x)
	#this function further builds the graph by adding the required loss operations(ops).
	#this averages the cross entropy values across the batch dimension (the first dimension) as the total loss.
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	#This function adds the operations needed to minimize the loss
	#this uses adams optimizer
	optimizer = tf.train.AdamOptimizer().minimize(cost) #learning_rate = 0.001
	#total number of iterations/cycles to run.
	epochs = 15
	#starting a session. 
	#this runs the model and gives the output.
	with tf.Session() as sess:
		#initializing the variables.
		sess.run(tf.initialize_all_variables())
		#Training loop.
		for epoch in range(epochs):
			epoch_loss = 0
			i = 0
			#training in batches.
			while i < len(trainx):
				start = i
				end = i+batch_size
				batchx = np.array(trainx[start:end])
				batchy = np.array(trainy[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x:batchx, y:batchy})
				epoch_loss += c
				i+=batch_size
			print ('Epoch', epoch+1, '/', epochs, 'loss', epoch_loss)
		#evaluating the model.
		#argmax: Returns the index with the largest value across dimensions of a tensor
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:testx, y:testy}))

#calling the function to train.
trainNN(x)
