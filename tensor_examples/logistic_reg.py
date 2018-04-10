import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from pylab import rand,plot,show,norm

def get_linearly_separable_data():
	separable = False
	while not separable:
		samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
		red = samples[0][samples[1] == 0]
		blue = samples[0][samples[1] == 1]
		separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
	x_train = np.array(samples[0])
	y_train = np.array(samples[1])
	plt.plot(red[:, 0], red[:, 1], 'r.')
	plt.plot(blue[:, 0], blue[:, 1], 'b.')
	plt.show()
	
	return x_train,y_train.reshape(-1,1) ,red ,blue
	 
	 
# PLACEHOLDERS
x = tf.placeholder(tf.float32, [None , 2])
y = tf.placeholder(tf.float32, [None,1])
# MODEL
W = tf.Variable(tf.truncated_normal([2, 1], mean=0.0, stddev=1.0))
b = tf.Variable(tf.zeros(1))
output = tf.add(tf.matmul(x, W), b)
sig = tf.sigmoid(output)#activation 
#COST
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits  = sig, labels = y)
#OPTIMIZER 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

# GENERATING DATA 
train_inputs , train_targets , red ,blue  = get_linearly_separable_data()
# SESSION
epochs = 5000
with tf.Session () as sess: 
	sess.run(tf.initialize_all_variables()) 
	for epoch in range(epochs): 
		
		sess.run(train_op, feed_dict={
                 x: train_inputs,
                 y: train_targets,
                  })
				 
		learned_variables = sess.run([W,b],feed_dict={
                 x: train_inputs,
                 y: train_targets,
                  }) 
	print ("variables :\n  w = {},\n b = {}".format(learned_variables[0] , learned_variables[1]) )
	###just sample 2 points on the line in order to draw it	
	y1 = (-2.5 * learned_variables[0][0] +  learned_variables[1]) / -learned_variables[0][1] 
	y2 = (2.5 * learned_variables[0][0] +  learned_variables[1]) / -learned_variables[0][1] 

	plt.plot(red[:, 0], red[:, 1], 'r.')
	plt.plot(blue[:, 0], blue[:, 1], 'b.')
	n = norm(learned_variables[0])
	plot([-2.5 ,2.5],[y1, y2],'--k')

	show()
   