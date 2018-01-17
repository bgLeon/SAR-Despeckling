import tensorflow as tf
import sys
import numpy as np
import math
import time
from tensorflow.python import debug as tf_debug
from cnn_utils import *
from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

path='datasets/SpeckleBSD.h5'
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset(path)
Deepness=17 #to change the number of layers

def create_placeholders(n_H0,n_W0,c, n_HY, n_WY):
	with tf.device('/device:GPU:0'):
		X=tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,c),name="X_train")
		Y=tf.placeholder(tf.float32, shape=(None,n_HY,n_WY,c),name="Y_train")
	return X,Y

def initialize_parameters(Nlayers,c):
	with tf.device('/device:GPU:0'):
		seed=10
		W1=tf.get_variable("W1", [3,3,c,64], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
		b1=tf.Variable(tf.constant(0.1, shape=[64]),name="b1")
		parameters = {}
		parameters["W1"]=W1
		parameters["b1"]=b1
		for l in range (2,Nlayers):
			parameters["W"+str(l)]=tf.get_variable("W"+str(l),[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer(seed=seed))
			parameters["b"+str(l)]=tf.Variable(tf.constant(0.1, shape=[64]),name="b"+str(l))
		parameters["W"+str(Nlayers)]=tf.get_variable("W"+str(Nlayers),[3,3,64,c],initializer=tf.contrib.layers.xavier_initializer(seed=seed))
		parameters["b"+str(Nlayers)]=tf.Variable(tf.constant(0.1, shape=[c]),"b"+str(Nlayers))
	return parameters

def forward_prop(X,parameters,Nlayers):
	outputs= {}
	# Conv+Relu
	with tf.name_scope("Conv_Relu"):
		with tf.device('/device:GPU:0'):
			Z1=tf.nn.conv2d(tf.log(X),parameters['W1'],strides=[1,1,1,1], padding='SAME',name="Z1")+parameters['b1']
			outputs["Z1"]=tf.nn.relu(Z1,name="A1")

			tf.summary.histogram("weights",parameters['W1'])
			tf.summary.histogram("biases",parameters['b1'])
			tf.summary.histogram("activations",outputs["Z1"])
	# Conv + BN + ReLU
	with tf.name_scope("Conv_BN_Relu"):
		for l in range (2,Nlayers):
			with tf.name_scope("Conv"+str(l)):
				with tf.device('/device:GPU:0'):
					Z=tf.nn.conv2d(outputs["Z"+str(l-1)],parameters['W'+str(l)],strides=[1,1,1,1], padding='SAME',name="Z"+str(l))
					BN=tf.contrib.layers.batch_norm(Z+parameters['b'+str(l)],fused=True)
					outputs["Z"+str(l)]=tf.nn.relu(BN,name="A"+str(l))
					tf.summary.histogram("weights",parameters['W'+str(l)])
					tf.summary.histogram("biases",parameters['b'+str(l)])
					tf.summary.histogram("activations",outputs["Z"+str(l)])
	# Conv
	with tf.name_scope("Conv"):
		with tf.device('/device:GPU:0'):	
			outputs["Z"+str(Nlayers)]=tf.nn.conv2d(outputs["Z"+str(Nlayers-1)],
				parameters['W'+str(Nlayers)],strides=[1,1,1,1], padding='SAME',name="Z"+str(Nlayers))+parameters['b'+str(Nlayers)]
			tf.summary.histogram("weights",parameters['W'+str(Nlayers)])
	return outputs["Z"+str(Nlayers)]


# cost function, mean is there in case you have a Speckle non-zero mean
def compute_cost (Zout,X,Y,mean):
	with tf.device('/device:GPU:0'):
		Mean=tf.Variable(mean,name="mean",dtype=tf.float32)
		cost=tf.reduce_mean(tf.log(tf.cosh(tf.subtract(tf.add(Zout,Mean),tf.log(np.divide(X,Y),name="Log1")
		,name="Subtract"),name="Cosh"),name="Log2"),name="cost")
		tf.summary.scalar("cost",cost)
	return cost

def compute_psnr(X_proc,Zout_proc,Orig_IMG_proc):
	Cleaned_IMG_proc=tf.subtract(tf.log(X_proc),Zout_proc)
	X=tf.clip_by_value(X_proc-1,0.0,255.0)
	Zout=tf.clip_by_value(tf.exp(Zout_proc)-1,0.0,255.0)
	Orig_IMG=tf.clip_by_value(Orig_IMG_proc-1,0.0,255.0)
	Cleaned_IMG=tf.clip_by_value(tf.exp(Cleaned_IMG_proc)-1,0.0,255.0)
	mse = tf.reduce_mean(tf.square(tf.subtract(Cleaned_IMG,Orig_IMG)),name="mse")
	if mse == 0.0:	#to avoid dividing by 0
		return 100
	PIXEL_MAX = 255.0
	PSNR=tf.multiply(tf.constant(20, dtype=tf.float32),tflog10(tf.divide(PIXEL_MAX,tf.sqrt(mse))),name="PSNR")
	# tf.summary.histogram(Orig_Noise)
	tf.summary.image("4_GT",Orig_IMG, 1)
	tf.summary.image("1_Input",X, 1)
	tf.summary.image("2_Noise",Zout, 1)
	tf.summary.image("3_Cleaned_IMG",Cleaned_IMG, 1)
	
	tf.summary.scalar("PSNR",PSNR)
	return PSNR

def model(X_train,Y_train,X_test,Y_test,
	num_epochs=55, minibatch_size=256, print_cost=True):
	ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
	seed=3
	X_train,Y_train,X_test,Y_test=prepare_images(X_train,Y_train,X_test,Y_test)
	(m,n_H0,n_W0,c)=X_train.shape
	(m,n_HY,n_WY,c)=Y_train.shape
	num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
	print("num_minibatches= "+str(num_minibatches))
	global_step = tf.Variable(0, trainable=False)
	boundaries = [27, 30] #when do we change learning rate
	values = [0.001, 0.0005, 0.0001]
	learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
	PSNRs=[]
	costs=[]
	with tf.device('/device:GPU:0'):
		X, Y = create_placeholders(n_H0, n_W0, c, n_HY,n_WY)
		parameters=initialize_parameters(Deepness,c)
		Zout=forward_prop(X,parameters,Deepness)
		with tf.name_scope("loss"):
				cost=compute_cost(Zout,X,Y,0)
		with tf.name_scope("train"):
				optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
		with tf.name_scope("PSNR"):
				psnr=compute_psnr(X,Zout,Y)

	# Initialize all the variables globally
		init = tf.global_variables_initializer()
	# Start the session to compute the tensorflow graph
	with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement = True)) as sess:
	# Run the initialization
		with tf.device('/device:GPU:0'):
			merged_summary = tf.summary.merge_all()
			writer = tf.summary.FileWriter("./Board/BSD/11")
			writer.add_graph(sess.graph)
			sess.run(init) 
			# Do the training loop
			for epoch in range(num_epochs):
				minibatch_cost=0
				minibatch_psnrs=[]
				seed=seed+1
				minibatches=random_mini_batches(X_train,Y_train,c,minibatch_size,seed)

				for minibatch in minibatches:
					# Select a minibatch
					(minibatch_X, minibatch_Y) = minibatch
					with tf.device('/device:GPU:0'):
						s,_ , temp_cost, temp_psnr = sess.run([merged_summary,optimizer,cost,psnr], feed_dict={X:minibatch_X, Y:minibatch_Y})
						# if epoch%3==0 or epoch == 50:
						writer.add_summary(s)
						minibatch_cost += temp_cost / num_minibatches#cambiar
						minibatch_psnrs.append(temp_psnr)
			   	# Print the cost every 5 epoch
				if print_cost == True and (epoch % 3 == 0 or epoch == 50):
					print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
					print ("PSNR after epoch %i: %f" % (epoch, np.mean(minibatch_psnrs)))

		#PSNR metrics
		minibatches_training=random_mini_batches(X_train,Y_train,c,minibatch_size,1)
		minibatches_testing=random_mini_batches(X_test,Y_test,c,minibatch_size,1)
		accuracy_PSNR = compute_psnr(X,Zout,Y)
		training=[]
		testing=[]
		for batch in minibatches_training:
			(minibatch_X, minibatch_Y) = batch
			temp_train_PSNR_accuracy = accuracy_PSNR.eval({X: minibatch_X, Y: minibatch_Y})
			training.append(temp_train_PSNR_accuracy)
		startTime=time.time()
		for batch in minibatches_testing:
			(minibatch_X, minibatch_Y) = batch
			temp_test_PSNR_accuracy = accuracy_PSNR.eval({X: minibatch_X, Y: minibatch_Y})
			testing.append(temp_test_PSNR_accuracy)
		train_PSNR_accuracy=np.mean(training)
		test_PSNR_accuracy=np.mean(testing)
		print("Train Accuracy:", train_PSNR_accuracy)
		print("Test Accuracy:", test_PSNR_accuracy)
	print("Time taken: %f seg" % ((time.time() - startTime)))
	return parameters
parameters = model(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
