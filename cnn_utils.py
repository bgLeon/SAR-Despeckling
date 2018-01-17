import math
import numpy as np
import h5py
import tensorflow as tf

def load_dataset(path):
	dataset=h5py.File(path,'r')

	train_set_x_orig=np.array(dataset['/training/noisy'][:])
	train_set_y_orig=np.array(dataset['/training/original'][:])
	test_set_x_orig=np.array(dataset['testing/noisy'][:])
	test_set_y_orig=np.array(dataset['/testing/original'][:])

	train_set_x_orig=np.rollaxis(train_set_x_orig,2,1)
	train_set_y_orig=np.rollaxis(train_set_y_orig,2,1)
	test_set_x_orig=np.rollaxis(test_set_x_orig,2,1)
	test_set_y_orig=np.rollaxis(test_set_y_orig,2,1)

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

# Creates a list of random minibatches from (X, Y)
def random_mini_batches(X,Y,c,mini_batch_size = 64, seed= 0):

	m=X.shape[0]
	mini_batches=[]
	np.random.seed(seed)

	#Shuffle (X,Y)
	permutation=list(np.random.permutation(m))

	if c>1:
		shuffled_X=X[permutation,:,:,:]
		shuffled_Y=Y[permutation,:,:,:]
	else:
		shuffled_X=X[permutation,:,:]
		shuffled_Y=Y[permutation,:,:]
	#Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches=math.floor(m/mini_batch_size)
	if c>1:
		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size,:,:,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size,:,:,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		# Handling the end case (last mini-batch < mini_batch_size)
		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
			mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:,:,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
	else:
		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size,:,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size,:,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		# Handling the end case (last mini-batch < mini_batch_size)
		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:]
			mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
	return mini_batches
def tflog10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return tf.divide(numerator,denominator)
def prepare_images(X_train,Y_train,X_test,Y_test):
	X_train = X_train.astype(np.uint16)+1#/255
	X_test = X_test.astype(np.uint16)+1#/255
	Y_train = Y_train.astype(np.uint16)+1#/255.
	Y_test = Y_test.astype(np.uint16)+1#/255.
	try:
 		(m,n_H0,n_W0,c)=X_train.shape
 		(m,n_HY,n_WY,c)=Y_train.shape
 		(m,n_HY,n_WY,c)=X_test.shape
 		(m,n_HY,n_WY,c)=Y_test.shape
	except:
		(m,n_H0,n_W0)=X_train.shape
		(m,n_HY,n_WY)=Y_train.shape
		print("C=1, no color")
		X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
		Y_train=np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
		X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
		Y_test=np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))
	return X_train,Y_train,X_test,Y_test