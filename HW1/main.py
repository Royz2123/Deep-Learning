'''
Some 1 and 2 layer networks written using Python2.7 and Tensorflow
This example is using the MNIST database of handwritten digits
Authors: Roy Zohar and Roy Mezan
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
number_of_neurons = 256
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
# None because we don't know how many inputs
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# MAIN FUNCTION

# Main in the same format that as asked
def main():
    number_of_neurons = 256
    one_hidden_layer_no_activation(number_of_neurons)
    two_hidden_layers_no_activation(number_of_neurons)
    two_hidden_layers_sigmoid(number_of_neurons)
    two_hidden_layers_relu(number_of_neurons)


# HELPER FUNCTIONS:

# Create model
def multilayer_perceptron(
    x,
    weights,
    biases,
    activation,
    layers=2,
):
    layers = []

    # Hidden layer 1
    layers.append(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layers[0] = activation(layers[0])

    # Hidden layer 2
    if layers == 2:
        layers.append(tf.add(tf.matmul(layers[0], weights['h2']), biases['b2']))
        layers[1] = activation(layers[1])

    # Output layer with no activation
    return tf.matmul(layers[-1], weights['out']) + biases['out']


# Store layers weight & bias
# initializes the weights and biases. Assumes that all hidden layers have the
# same amount of neurons
def init_weights(
    layers=2,
    inp=n_input,
    hid=number_of_neurons,
    out=n_classes
):
    weights = {
        'h1': tf.Variable(tf.random_normal([inp, hid])),
        'out': tf.Variable(tf.random_normal([hid, out]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([hid])),
        'out': tf.Variable(tf.random_normal([out]))
    }

    # Create more vars if more layers
    # currently only handling 2 layers max
    if layers==2:
        weights['h2'] = tf.Variable(tf.random_normal([hid, hid]))
        biases['b2'] = tf.Variable(tf.random_normal([hid]))

    return (weights, biases)


# Run the tensorflow graph
def run_graph(
    init,
    optimizer,
    cost,
    pred
):
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



def one_hidden_layer_no_activation(number_of_neurons):
    weights, biases = init_weights(1, hid=number_of_neurons)
    pred = multilayer_perceptron(
        x,
        weights,
        biases,
        lambda _: _,       # No activation
        layers=1,
    )

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    run_graph(init, optimizer, cost, pred)


def two_hidden_layers_no_activation(number_of_neurons):
    weights, biases = init_weights(layers=2, hid=number_of_neurons)
    pred = multilayer_perceptron(
        x,
        weights,
        biases,
        lambda _: _,       # No activation
        layers=2,
    )

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    run_graph(init, optimizer, cost, pred)


def two_hidden_layers_sigmoid(number_of_neurons):
    weights, biases = init_weights(layers=2, hid=number_of_neurons)
    pred = multilayer_perceptron(
        x,
        weights,
        biases,
        tf.nn.sigmoid,       # Sigmoid activation
        layers=2,
    )

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    run_graph(init, optimizer, cost, pred)


def two_hidden_layers_relu(number_of_neurons):
    weights, biases = init_weights(layers=2, hid=number_of_neurons)
    pred = multilayer_perceptron(
        x,
        weights,
        biases,
        tf.nn.relu,       # RElu activation
        layers=2,
    )

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    run_graph(init, optimizer, cost, pred)



if __name__ == '__main__':
	main()
