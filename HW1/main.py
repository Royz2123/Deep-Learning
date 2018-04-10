import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# consts

in_shape = 28 * 28
out_shape = 10

learning_rate = 0.001
epochs = 15
batch_size = 100


def fully_connected(input, weights, biases, activations):
	layers = []
	layers.append(input)
	for w, b, ac in zip(weights, biases, activations):
		layer = ac(tf.matmul(layers[-1], w) + b)
		layers.append(layer)
	
	return layers


def one_hidden_layer_no_activation(number_of_neurons):
	x = tf.placeholder(tf.float32, [None, in_shape])
	y = tf.placeholder(tf.float32, [None, out_shape])
	
	weights = [tf.Variable(tf.random_normal([in_shape, number_of_neurons])),
			   tf.Variable(tf.random_normal([number_of_neurons, out_shape]))]
	
	biases = [tf.Variable(tf.random_normal([number_of_neurons])),
			  tf.Variable(tf.random_normal([out_shape]))]
	
	activations = [lambda _: _] * 2
	
	layers = fully_connected(x, weights, biases, activations)
	
	out = layers[-1]
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	var_init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(var_init)
		
		for epoch in range(epochs):
			total_cost = 0
			batch_num = int(mnist.train.num_examples / batch_size)
			for i in range(batch_num):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				feed_dict = {x: batch_x,
							 y: batch_y}
				_, c = sess.run([optimizer, cost], feed_dict=feed_dict)
				total_cost += c
			
			total_cost /= batch_num
			print('Epoch: ' + str(epoch) + ', cost: ' + str(total_cost))
		
		# testing
		corrects = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
		print('acc: ' + str(acc.eval({x: mnist.test.images, y: mnist.test.labels})))


def two_hidden_layers_no_activation(number_of_neurons):
	x = tf.placeholder(tf.float32, [None, in_shape])
	y = tf.placeholder(tf.float32, [None, out_shape])
	
	weights = [
		tf.Variable(tf.random_normal([in_shape, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, out_shape]))
	]
	
	biases = [
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([out_shape]))
	]
	
	activations = [lambda _: _] * 3
	
	layers = fully_connected(x, weights, biases, activations)
	
	out = layers[-1]
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	var_init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(var_init)
		
		for epoch in range(epochs):
			total_cost = 0
			batch_num = int(mnist.train.num_examples / batch_size)
			for i in range(batch_num):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				feed_dict = {x: batch_x,
							 y: batch_y}
				_, c = sess.run([optimizer, cost], feed_dict=feed_dict)
				total_cost += c
			
			total_cost /= batch_num
			print('Epoch: ' + str(epoch) + ', cost: ' + str(total_cost))
		
		# testing
		corrects = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
		print('acc: ' + str(acc.eval({x: mnist.test.images, y: mnist.test.labels})))


def two_hidden_layers_sigmoid(number_of_neurons):
	x = tf.placeholder(tf.float32, [None, in_shape])
	y = tf.placeholder(tf.float32, [None, out_shape])
	
	weights = [
		tf.Variable(tf.random_normal([in_shape, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, out_shape]))
	]
	
	biases = [
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([out_shape]))
	]
	
	activations = [tf.nn.sigmoid] * 2 + [lambda _: _]
	
	layers = fully_connected(x, weights, biases, activations)
	
	out = layers[-1]
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	var_init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(var_init)
		
		for epoch in range(epochs):
			total_cost = 0
			batch_num = int(mnist.train.num_examples / batch_size)
			for i in range(batch_num):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				feed_dict = {x: batch_x,
							 y: batch_y}
				_, c = sess.run([optimizer, cost], feed_dict=feed_dict)
				total_cost += c
			
			total_cost /= batch_num
			print('Epoch: ' + str(epoch) + ', cost: ' + str(total_cost))
		
		# testing
		corrects = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
		print('acc: ' + str(acc.eval({x: mnist.test.images, y: mnist.test.labels})))


def two_hidden_layers_relu(number_of_neurons):
	x = tf.placeholder(tf.float32, [None, in_shape])
	y = tf.placeholder(tf.float32, [None, out_shape])
	
	weights = [
		tf.Variable(tf.random_normal([in_shape, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons, out_shape]))
	]
	
	biases = [
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([number_of_neurons])),
		tf.Variable(tf.random_normal([out_shape]))
	]
	
	activations = [tf.nn.relu] * 2 + [lambda _: _]
	
	layers = fully_connected(x, weights, biases, activations)
	
	out = layers[-1]
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	var_init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(var_init)
		
		for epoch in range(epochs):
			total_cost = 0
			batch_num = int(mnist.train.num_examples / batch_size)
			for i in range(batch_num):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				feed_dict = {x: batch_x,
							 y: batch_y}
				_, c = sess.run([optimizer, cost], feed_dict=feed_dict)
				total_cost += c
			
			total_cost /= batch_num
			print('Epoch: ' + str(epoch) + ', cost: ' + str(total_cost))
		
		# testing
		corrects = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
		print('acc: ' + str(acc.eval({x: mnist.test.images, y: mnist.test.labels})))


def main():
	number_of_neurons = 256
	
	one_hidden_layer_no_activation(number_of_neurons)
	two_hidden_layers_no_activation(number_of_neurons)
	two_hidden_layers_sigmoid(number_of_neurons)
	two_hidden_layers_relu(number_of_neurons)


if __name__ == '__main__':
	main()