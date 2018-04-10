import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from pylab import rand,plot,show,norm


x = tf.placeholder(tf.float32, [None , 2])
y = tf.placeholder(tf.float32, [None,1])

number_of_nuerons = 4
W = tf.Variable(tf.random_uniform([2, 2] ,-1 ,1))
W2 =  tf.Variable(tf.random_uniform([2, 1] ,-1 ,1))
b = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

layer1 = tf.sigmoid(tf.add(tf.matmul(x, W), b))
layer2 =tf.sigmoid(tf.add(tf.matmul(layer1, W2) , b2))

cost = tf.reduce_mean(((y*tf.log(layer2))  + ((1-y)*tf.log(1.0 - layer2))) *-1 )

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(cost)

train_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
train_targets =np.array( [0,1,1,0]).reshape(-1,1)
epochs = 150000
with tf.Session () as sess: 
	sess.run(tf.initialize_all_variables()) 
	for epoch in range(epochs): 
		sess.run(train_op, feed_dict={
                 x: train_inputs,
                 y: train_targets,
                  })
		if epoch%1000 == 0 :
			res =  sess.run(layer2,feed_dict={ x: train_inputs, y: train_targets})
			print ("epoch {} : ".format(epoch))
			for i,p in enumerate(train_inputs) :
				print ("====> xor for {} is {}".format(p,res[i]))
			
	print "result :{}".format(np.round(sess.run(layer2,feed_dict={ x: train_inputs, y: train_targets})))