import numpy as np
from numpy.linalg import inv
from numpy import linalg
import math
from random import randint
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

# Q1 - Read data from the input file, and also encode the first column as three column binary
def read_data (input_file):
	file = open(input_file, 'r')
	lines_list = file.readlines()
	lines_list = [i.rstrip() for i in lines_list]

	X = []
	Y = []
	for line in lines_list:
		col = [1]
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

def ridgeerror(X, Y, W, lamda):
	N = X.shape[0]
	diff = np.dot(X, W) - Y
	SE = np.dot (np.transpose(diff), diff)/(2*N)
	reg_err = (lamda/2) * linalg.norm(W, 2)
	total_error = SE + reg_err
	return total_error


# Q3 - My linear ridge regression
def mylinridgereg(X, Y, lamda):

	if (lamda!=0):
		# print("Analytical solution")
		# Analytical Solution
		temp = ( np.dot(np.transpose(X), X) + ( lamda* np.identity(X.shape[1]) ) )
		# print("Shape of temp = " + str(temp.shape) )
		# print("Determinant of temp = " + str(np.linalg.det(temp) ) )
		W =  np.dot( inv(temp) , np.dot(np.transpose(X), Y) ) ;
		return W

	else :
	# Gradient descent
		W = np.zeros(X.shape[1])
		N = X.shape[0]
		err = ridgeerror(X,Y,W,lamda)
		err_prev = err + 100
		learning_rate = 0.01
		while (err_prev - err) > 0.00001 :
			# print(err)
			gradient = np.dot(np.transpose(X), (np.dot(X,W)-Y))/N + lamda*W
			W = W - (learning_rate* gradient)
			err_prev = err
			err = ridgeerror(X,Y,W,lamda)
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
	return err/(2*actual.shape[0])

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



# -----------------------------------------------------------------------------


# Script

if len(sys.argv)!=2:
	print("The correct format for running code is -> python Q1.py question_number")

question = int(sys.argv[1])
if (question<1 and question>10):
	print ("The question number should be between 1 and 10")


# Q1 - Reading and encoding data
if (question == 1):
	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	print("Question 1:\nRead data from file and encoded the first argument as required")
	print("\nAttributes:")
	print(X_original)	
	print("\nTarget Values:")
	print(Y_original)	
# ------------------------------------------------------------------------

# Q2 - Standardizing the independent variables
elif (question == 2):
	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	X = np.array(X_original)
	Y = np.array(Y_original)
	X[:, 4:] = standardize_data(X_original[:, 4:])
	print("Question 2:\nStandardizing atributes, we standardize 4:11 atributes")
	print("\nStandardized Attributes:")
	print(X)	

# ------------------------------------------------------------------------

#Q3,Q4,Q5 - Linear ridge regression and measuring error for variety of lambdas
elif (question == 3 or question == 4 or question == 5):
	print("Question 3,4,5:\nPartitioning data, applying regression, evaluating and measuring MSE\n")
	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	X = np.array(X_original)
	Y = np.array(Y_original)
	X[:, 4:] = standardize_data(X_original[:, 4:])
	(X_training, Y_training, X_test, Y_test) = partition_data(0.8, X, Y)

	lambdas = list(range(0,21))
	lambdas = [float(i)*0.2 for i in lambdas]
	for lamda in lambdas:
		W = mylinridgereg(X_training, Y_training, lamda)
		Y_training_predicted = mylinridgeregeval(X_training, W)
		Y_test_predicted = mylinridgeregeval(X_test, W)
		print ("Lambda value = " + str(lamda) + ", training partition fraction = "+ str(0.8)  )
		print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
		print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )


# -----------------------------------------------------------------------------


# Q6
elif (question == 6):
	lamda = 0.1
	print("Question 6")
	print ("Lambda value = " + str(lamda)  )
	print("\nBefore removing any attributes")

	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	X = np.array(X_original)
	Y = np.array(Y_original)
	X[:, 4:] = standardize_data(X_original[:, 4:])
	(X_training, Y_training, X_test, Y_test) = partition_data(0.8, X, Y)

	W = mylinridgereg(X_training, Y_training, lamda)
	Y_training_predicted = mylinridgeregeval(X_training, W)
	Y_test_predicted = mylinridgeregeval(X_test, W)
	print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
	print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )

	print("\nAfter removing 2 least significant attributes")
	(X_training, X_test) = remove_least_significant_attributes (X_training, X_test, Y_training, lamda)
	W = mylinridgereg(X_training, Y_training, lamda)
	Y_training_predicted = mylinridgeregeval(X_training, W)
	Y_test_predicted = mylinridgeregeval(X_test, W)
	print ( "Training set mean square error = " + str(meansquarederr(Y_training_predicted, Y_training)) )
	print ( "Test set mean square error = " + str(meansquarederr(Y_test_predicted, Y_test)) + "\n" )



# ----------------------------------------------------------------------------------------------------


#  Question 7,8,9
elif (question == 7 or question == 8 or question == 9):
	print ("Question 7,8,9\n")
	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	average_test_MSE = []
	average_train_MSE = []

	fracs = list(range(1,10))
	fracs = [float(i)*0.1 for i in fracs]
	lambdas = list(range(0,10))
	lambdas = [float(i)*0.4 for i in lambdas]


	for frac in fracs:
		test_col = []
		train_col = []
		for lamda in lambdas:
			print("\nFraction = " + str(frac) + ", Lambda = " + str(lamda) + " : ")
			test_err = 0
			train_err = 0
			for j in range(0,100):
				X = np.array(X_original)
				Y = np.array(Y_original)
				(X_training, Y_training, X_test, Y_test) = partition_data(frac, X, Y)
				X_std = X_training[:, 4:]
				mean = np.mean(X_std, axis=0)
				std_dev = np.std(X_std, axis=0);
				X_training[:, 4:] = X_training[:, 4:]-mean
				X_training[:, 4:] = X_training[:, 4:]/std_dev
				X_test[:, 4:] = X_test[:, 4:]-mean
				X_test[:, 4:] = X_test[:, 4:]/std_dev

				W = mylinridgereg(X_training, Y_training, lamda)
				Y_training_predicted = mylinridgeregeval(X_training, W)
				Y_test_predicted = mylinridgeregeval(X_test, W)
				test_err += meansquarederr(Y_test_predicted, Y_test)
				train_err += meansquarederr(Y_training_predicted, Y_training)

			test_err /= 100
			train_err /= 100
			print("Average test MSE ", test_err)
			print("Average train MSE ", train_err)
			test_col.append(test_err)
			train_col.append(train_err)

		ax = plt.subplot(3,3, int(frac/0.1))

		ax.axis([0, 4, 4.6, 5.4])
		ax.set_title('frac = ' + str(frac))
		ax.set_xlabel('Lambda')
		ax.set_ylabel('Mean square error')
		test_line, = ax.plot(lambdas, test_col, 'r-', label="Test set")
		train_line = ax.plot(lambdas, train_col, 'b-', label = "Training set")

		average_test_MSE.append(test_col)
		average_train_MSE.append(train_col)

	plt.tight_layout()
	plt.savefig("Q8.jpg")
	plt.show()
	plt.close()

	min_ = np.min(average_test_MSE, axis=1)
	plt.xlabel('Training fraction')
	plt.ylabel('Mean squared error')
	plt.title('Minimum average mean squared error on test data for different training fractions')
	plt.axis([0, 1, 4.6, 5.4])
	test_line, = plt.plot(fracs, min_, 'r-')
	plt.savefig("Q9_1.jpg")
	plt.show()
	plt.close()

	argmin_ = np.argmin(average_test_MSE, axis=1)
	argmin_ = argmin_ * 0.4
	plt.xlabel('Training fraction')
	plt.ylabel('Lambda')
	plt.title('Lamda for which average MSE is mininum for different training fractions')
	plt.axis([0, 1, 0, 5])
	test_line, = plt.plot(fracs, argmin_, 'r-')
	plt.savefig("Q9_2.jpg")
	plt.show()
	plt.close()


# -----------------------------------------------------------------------------


# Question 10
elif (question == 10):
	print ("Question 10\n")

	data_file = "linregdata"
	(X_original, Y_original) = read_data(data_file)
	frac = 0.8
	lamda = 1.6
	X = np.array(X_original)
	Y = np.array(Y_original)
	(X_training, Y_training, X_test, Y_test) = partition_data(frac, X, Y)
	X_std = X_training[:, 4:]
	mean = np.mean(X_std, axis=0)
	std_dev = np.std(X_std, axis=0);
	X_training[:, 4:] = X_training[:, 4:]-mean
	X_training[:, 4:] = X_training[:, 4:]/std_dev
	X_test[:, 4:] = X_test[:, 4:]-mean
	X_test[:, 4:] = X_test[:, 4:]/std_dev

	W = mylinridgereg(X_training, Y_training, lamda)
	Y_training_predicted = mylinridgeregeval(X_training, W)
	Y_test_predicted = mylinridgeregeval(X_test, W)

	plt.xlabel('Predicted target value')
	plt.ylabel('Actual target value')
	plt.title('Actual and predicted target values')
	plt.axis([0, 30, 0, 30])
	xyline, = plt.plot([0, 30], [0, 30], 'g-', label="x=y line")
	test_line, = plt.plot(Y_test_predicted, Y_test, 'ro', alpha=0.1, label="Test set")
	train_line = plt.plot(Y_training_predicted, Y_training, 'bo', alpha=0.1, label = "Training set")
	plt.legend()
	plt.savefig('Q10.jpg')
	plt.show()
	plt.close()


# -----------------------------------------------------------------------------


