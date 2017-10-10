import numpy as np
from numpy.linalg import inv
import math
from random import randint


# Q1 - Read data from the input file, and also encode the first column as three column binary
def read_data (input_file):
	file = open(input_file, 'r')
	lines_list = file.readlines()
	lines_list = [i.rstrip() for i in lines_list]

	X = []
	Y = []
	for line in lines_list:
		col = []
		line_split = line.split(',')
		if (line_split[0] == 'F'):
			col.append(1)
			col.append(0)
			col.append(0)
		elif (line_split[0] == 'I'):
			col.append(0)
			col.append(1)
			col.append(0)
		else:
			col.append(0)
			col.append(0)
			col.append(1)
		col.append( float(line_split[1]) )
		col.append( float(line_split[2]) )
		col.append( float(line_split[3]) )
		col.append( float(line_split[4]) )
		col.append( float(line_split[5]) )
		col.append( float(line_split[6]) )
		col.append( float(line_split[7]) )
		X.append(col);
		Y.append( float(line_split[8]) )

	X = np.array(X)
	Y = np.array(Y)
	return (X,Y)

# Q2
def standardize_data (X):
	mean = np.mean(X, axis=0)
	var = np.var(X, axis=0);
	std_dev = [math.sqrt(i) for i in var]
	X = X-mean;
	X = X/std_dev;
	return X

# Q3 - My linear ridge regression
def mylinridgereg(X, Y, lamda):
	W =  np.dot( inv( np.dot(np.transpose(X), X) + ( lamda* np.identity(X.shape[1]) ) ) , np.dot(np.transpose(X), Y) ) ;
	return W

# Q3 - Prediction of target variable 
def mylinridgeregeval(X, weights):
	Y_predicted = np.dot(X, weights);
	return Y_predicted

# Q4
def partition_data(frac, X, Y):
	N = Y.shape[0]
	for i in range(0, N):
		r = randint(0, N-1)
		temp = X[i]
		X[i] = X[r]
		X[r] = temp
		temp2 = Y[i]
		Y[i] = Y[r]
		Y[r] = temp2

	no_training = math.floor(frac*N)
	X_training = X[0:no_training+1]
	Y_training = Y[0:no_training+1]
	X_test = X[no_training+1:N,]
	Y_test = Y[no_training+1:N]
	return (X_training, Y_training, X_test, Y_test)


# Q5
def meansquarederr(predicted, actual):
	diff = actual - predicted
	diff = [i**2 for i in diff]
	err = sum(diff)
	return err

# Q6
def remove_least_significant_attributes (X_training, X_test, Y_training, lamda):
	W = mylinridgereg(X_training, Y_training, lamda)
	abs_W = [abs(i) for i in W]

	# Deleting two least significant attributes
	index = abs_W.index(min(abs_W));
	abs_W = np.delete(abs_W, index, 0)
	W = np.delete(W, index, 0)
	X_training = np.delete(X_training, index, 1)
	X_test = np.delete(X_test, index, 1)

	index = abs_W.tolist().index(min(abs_W));
	abs_W = np.delete(abs_W, index, 0)
	W = np.delete(W, index, 0)
	X_training = np.delete(X_training, index, 1)
	X_test = np.delete(X_test, index, 1)

	return (X_training, X_test)




# Script
# Q1 - Reading and encoding data
data_file = "linregdata"
(X,Y) = read_data(data_file)

# Q2 - Standardizing the independent variables
# X = standardize_data(X)
# Checking standadization
# new_mean = np.mean(X, axis=0)
# new_var = np.var(X, axis=0);
# print(new_mean)
# print(new_var)

# Q4 - Paritioning the data
(X_training, Y_training, X_test, Y_test) = partition_data(0.8, X, Y)

#Q3,Q4,Q5 - Linear ridge regression and measuring error for variety of lambdas
lambdas = list(range(1,21))
lambdas = [float(i)*0.2 for i in lambdas]
for lamda in lambdas:
	W = mylinridgereg(X_training, Y_training, lamda)
	Y_training_predicted = mylinridgeregeval(X_training, W)
	Y_test_predicted = mylinridgeregeval(X_test, W)
	# print ("Lambda value = " + str(lamda)  )
	# print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
	# print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )


# Q6
lamda = 1.6
W = mylinridgereg(X_training, Y_training, lamda)
Y_training_predicted = mylinridgeregeval(X_training, W)
Y_test_predicted = mylinridgeregeval(X_test, W)
print ("Lambda value = " + str(lamda)  )
print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )

(X_training, X_test) = remove_least_significant_attributes (X_training, X_test, Y_training, lamda)
W = mylinridgereg(X_training, Y_training, lamda)
Y_training_predicted = mylinridgeregeval(X_training, W)
Y_test_predicted = mylinridgeregeval(X_test, W)
print ("Lambda value = " + str(lamda)  )
print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )


# Q7
