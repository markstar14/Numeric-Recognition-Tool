import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random
#%matplotlib inlinei
import pickle
# No update required !
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

# Updated and working fine
def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0 / (1.0 + np.exp(-1.0 * z))

def load_data(mat):
    # mat = loadmat('mnist_all_tiny.mat') #loads the MAT object as a Dictionary
    # added by : Zulkar
    # get all the training data:
    # print 'training data size:'
    train0 = mat.get('train0')
    # print train0.shape
    train1 = mat.get('train1')
    # print train1.shape
    train2 = mat.get('train2')
    # print train2.shape
    train3 = mat.get('train3')
    # print train3.shape
    train4 = mat.get('train4')
    # print train4.shape
    train5 = mat.get('train5')
    # print train5.shape
    train6 = mat.get('train6')
    # print train6.shape
    train7 = mat.get('train7')
    # print train7.shape
    train8 = mat.get('train8')
    # print train8.shape
    train9 = mat.get('train9')
    # print train9.shape

    # get all the testing data:
    # print 'testing data size:'
    test0 = mat.get('test0')
    # print test0.shape
    test1 = mat.get('test1')
    # print test1.shape
    test2 = mat.get('test2')
    # print test2.shape
    test3 = mat.get('test3')
    # print test3.shape
    test4 = mat.get('test4')
    # print test4.shape
    test5 = mat.get('test5')
    # print test5.shape
    test6 = mat.get('test6')
    # print test6.shape
    test7 = mat.get('test7')
    # print test7.shape
    test8 = mat.get('test8')
    # print test8.shape
    test9 = mat.get('test9')
    # print test9.shape

    ## (1) Stack all training matrices into one 60000  784 matrix. Do the same for test matrices.
    train_mat = np.concatenate((train0, train1, train2, train3, train4,
                            train5, train6, train7, train8, train9),
                            axis=0)

    test_mat = np.concatenate((test0, test1, test2, test3, test4,
                                test5, test6, test7, test8, test9),
                                axis=0)

    # training data: this array contains the size of each individual training data subsets .. train0, train1, ...
    shape_train = [train0.shape[0], train1.shape[0],train2.shape[0],train3.shape[0],
             train4.shape[0], train5.shape[0],train6.shape[0],train7.shape[0],
             train8.shape[0], train9.shape[0]]
    shape_train = np.array(shape_train)

    # test data :  this array contains the size of each individual test data subsets .. test0, test1, ...
    shape_test = [test0.shape[0], test1.shape[0],test2.shape[0],test3.shape[0],
             test4.shape[0], test5.shape[0],test6.shape[0],test7.shape[0],
             test8.shape[0], test9.shape[0]]

    return train_mat, test_mat, shape_train, shape_test


def generate_labels(shape_train, shape_test):


    training_data_list = []
    labels = 0;

    for shape in shape_train:
        array = np.ones((shape,), dtype=np.int)
        array = labels*array
        training_data_list.append(array)
        labels = labels + 1;

    train_label_all = []
    for classes in training_data_list:
        train_label_all = np.concatenate((train_label_all, classes), axis=0)


    test_data_list = []
    labels = 0;

    for shape in shape_test:
        array = np.ones((shape,), dtype=np.int)
        array = labels*array
        test_data_list.append(array)
        labels = labels + 1;

    test_label_all = []
    for classes in test_data_list:
        test_label_all = np.concatenate((test_label_all, classes), axis=0)

    return train_label_all, test_label_all

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat')
    #Pick a reasonable size for validation data
    validation_data_size = 10000

    #Your code here

    # added by: Zulkar
    max_pixel = 255.0

    train_mat, test_mat, shape_train, shape_test = load_data(mat)


    number_of_training_data = train_mat.shape[0]
    input_dimension = train_mat.shape[1]

    number_of_test_data = test_mat.shape[0]

    ##print"train_mat:"
    ##printtrain_mat
    ##print"test_mat:"
    ##printtest_mat

    ## (2) Create a 60000 length vector with true labels (digits) for each training example. Same for test data.
        # training data :

    train_label_all, test_label_all = generate_labels(shape_train, shape_test)
    ##print"train_label_all:"
    ##printtrain_label_all
    ##print"test_label_all:"
    ##printtest_label_all

    ## (3) Normalize the training matrix and test matrix so that the values are between 0 and 1.
    # training data :
    normalized_train_mat = (train_mat / max_pixel)
    ##print'normalized training matrix:'
    ##printnormalized_train_mat

    #test data :
    normalized_test_mat = (test_mat / max_pixel)
    ##print'normalized_test matrix:'
    ##printnormalized_test_mat


    ## (4) Randomly split the 60000 X 784 normalized matrix into two matrices:
    ## training matrix (50000 X 784) and
    ## validation matrix (10000 X 784).
    ## Make sure you split the true labels vector into two parts as well.

    # marge normalized training matrix and training label :
    new_size_train_mat_with_label = input_dimension + 1
    train_mat_with_label = np.zeros((number_of_training_data, new_size_train_mat_with_label))
    train_mat_with_label[:,:-1] = normalized_train_mat
    train_mat_with_label[:,-1] = train_label_all
    #print("train_mat_with_label:")
    #print(train_mat_with_label)
    #print('matrix dimension: ')
    #print(train_mat_with_label.shape)


    # shuffle rows randomly
    np.random.shuffle(train_mat_with_label)
    #print("training matrix with label after shuffle:")
    #print(train_mat_with_label)

    # perform split :
    size_of_training_data = number_of_training_data - validation_data_size

    train_data_all_features = train_mat_with_label[:size_of_training_data,:-1]
    train_label = train_mat_with_label[:size_of_training_data,-1]

    validation_data_all_features = train_mat_with_label[size_of_training_data:number_of_training_data,:-1]
    validation_label = train_mat_with_label[size_of_training_data:number_of_training_data,-1]

    validation_label = validation_label.astype(int)
    #print("Training data without feature extraction:")
    #print(train_data_all_features)
    #print("training data label:")
    #print(train_label)
    #print("validation data without feature extraction:")
    #print(validation_data_all_features)
    #print("validation label:")
    #print(validation_label)


    # test data :
    test_data_all_features = normalized_test_mat
    test_label = test_label_all.astype(int)
    #print("Test data without feature extraction:")
    #print(test_data_all_features)
    #print("Test data labels: ")
    #print(test_label)


    #    Feature selection : one can observe that there are many features which values are exactly the same for all data points in the training set.
    # we can ignore those features in the pre-processing step.
    # Observation : we can ignore the columns those have same values for all data points.

    # merge all the data to get homogenous feature space for all training, test and validation data
    all_data_all_features = np.concatenate((train_data_all_features,validation_data_all_features,test_data_all_features),axis = 0)

    #print("all data marged into a matrix: shape")
    ##printall_data_all_features
    #print(all_data_all_features.shape)

    temp_mat1 = all_data_all_features == all_data_all_features[0,:]
    #print "temp_mat1"
    #print temp_mat1
    temp_mat2 = np.all(temp_mat1, axis = 0)
    #print "temp_mat2"
    #print temp_mat2

    same_indices = []
    for i in range(temp_mat2.shape[0]):
        if temp_mat2[i]:
            same_indices.append(i)
    all_data = np.delete(all_data_all_features,same_indices,axis=1)
    #print "all_data:"
    #print all_data.shape
    #print all_data

    # split training , validation and tesing data:

    train_data = all_data[:size_of_training_data,:]
    #print("training data after feature selection:")
    #print(train_data)
    validation_data = all_data[size_of_training_data:number_of_training_data,:]
    #print("validation data after feature selection:")
    #print(validation_data)
    test_data = all_data[number_of_training_data:,:]
    #print("test data after feature selection:")
    #print(test_data)


    # commented by : Zulkar :
    # train_data = np.array([])
    # train_label = np.array([])
    # validation_data = np.array([])
    # validation_label = np.array([])
    # test_data = np.array([])
    # test_label = np.array([])

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    print('nnObjFunction')
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0


    # compute the feed forward pass for all the training input:
    #################################################################################
    # added by: Zulkar

    # Number of output nodes:
    #n_class = 10;

	# add bias 1 at (d+1) position of each data point
    N1 = training_data.shape[0]
    N2 = training_data.shape[1]
    train_data_bias = np.ones((N1,N2+1))
    train_data_bias[:,:-1] = training_data
    #print("train_data_bias")
    #print(train_data_bias.shape)
    #print(train_data_bias)

    #print("w1")
    #print(w1.shape)
    #print(w1)

	# computing dot product between data points and weights w1 : hidden layer
    a_hidden = np.inner(train_data_bias,w1)
    #print("a_hidden.shape")
    #print(a_hidden.shape)
    #print(a_hidden)

	# compute threshold function (sigmoid) : hidden Layer
    z_hidden = sigmoid(a_hidden)
    #print("z_hidden.shape")
    #print(z_hidden.shape)
    #print(z_hidden)

	# add bias hidden node (m+1)th to z_hidden. We set its value 1 directly
    N1 = z_hidden.shape[0]
    N2 = z_hidden.shape[1]
    z_hidden_bias = np.ones((N1,N2+1))
    z_hidden_bias[:,:-1] = z_hidden
    #print("z_hidden_bias")
    #print(z_hidden_bias.shape)
    #print(z_hidden_bias)

	# computing dot product between hidden layer output z_hidden_bias and weights w2
    a_output = np.inner(z_hidden_bias,w2)
    #print("a_output.shape : ")
    #print(a_output.shape )
    #print(a_output)

	# compute threshold function (sigmoid)  (equation 4)
    z_output = sigmoid(a_output)
    #print("z_output.shape")
    #print(z_output.shape)
    #print(z_output)


	#labels = np.argmax(z_output, axis=1)

	# added : zulkar:
	# creating training_label : converting each label to a 10 dimension vector.
    one_of_k = np.zeros((N1,n_class))
    i = 0
    for index in training_label:
        one_of_k[i,index] = 1
        i = i + 1
    #print("one_of_k.shape")
    #print(one_of_k.shape )
    #print(one_of_k)

	# compute the error function : J_p(W(1) ,W(2))  ... equation (5)
    difference =  one_of_k - z_output
    #print("difference.shape")
    #print(difference.shape)
    #print("difference")
    #print(difference)
    difference_squared = np.square(difference)
    summation_difference_squared = np.sum(difference_squared,axis = 1)
    Jp = 0.5 * summation_difference_squared

    #print("Jp")
    #print(Jp.shape)
    #print(Jp)

	# total error of the entire dataset : equation (6)
    number_of_training_data = training_data.shape[0]

    #print("number of training data is:")
    #print(number_of_training_data)

    J = sum(Jp)/number_of_training_data;  #  = 50000
    #print("J = ")
    #print(J)

	# compute the lambda error for output layer: equation (9)
    lambda_l_mat = (one_of_k - z_output)*(1- z_output)*z_output
    #print("lambda_l_mat")
    #print(lambda_l_mat.shape)
    #print(lambda_l_mat)

	# compute equation (7) :  derivative of error function
    sum_del_Jp_w2 = np.zeros( (lambda_l_mat.shape[1], z_hidden_bias.shape[1]))

    for i in range(lambda_l_mat.shape[0]):
        lambda_l = lambda_l_mat[i,:]
        z_j = z_hidden_bias[i,:]
        del_Jp_w2 = -1 * np.outer(lambda_l,z_j)
        sum_del_Jp_w2 = sum_del_Jp_w2 + del_Jp_w2
    #print("sum_del_Jp_w2")
    #print(sum_del_Jp_w2.shape)
    #print(sum_del_Jp_w2)

    def compute_lambda_l_into_weight_lj(lambda_l, w2_lj):
        #sum = 0
        sum = np.inner(lambda_l,w2_lj)
        return sum

	# computing equation (12) :


    del_Jp_w1 = np.zeros((n_hidden,n_input+1))  # 5x6
    sum_del_Jp_w1 = np.zeros((n_hidden,n_input+1))

    #print(del_Jp_w1.shape)

    #print("number of training data was:")
    #print(number_of_training_data)
    #print("n_hidden is:")
    #print(n_hidden)
    #print("n_input is")
    #print(n_input)

    # computing equation (12) :
    del_Jp_w1 = np.zeros((n_hidden,n_input+1))  # 5x6
    sum_del_Jp_w1 = np.zeros((n_hidden,n_input+1))
    #print del_Jp_w1.shape
    for p in range(number_of_training_data):
        sum_del_Jp_w1 = 0;
    #p = number_of_training_data - 1
        z_j = z_hidden_bias[p,:-1]

        lambda_l = lambda_l_mat[p,:]

        for j in range(n_hidden):  # 5
            w2_lj = w2[:,j]
            term = compute_lambda_l_into_weight_lj(lambda_l,w2_lj)
            for i in range(n_input+1):   # 6

                del_Jp_w1[j,i] = -1.0 * (1.0 - z_j[j]) * z_j[j] * term * train_data_bias[p,i]
        sum_del_Jp_w1 = sum_del_Jp_w1 + del_Jp_w1




    #print("sum_del_jp_w1.shape")
    #print(sum_del_Jp_w1.shape)
    #print(sum_del_Jp_w1)


	# compute equation (16):
    np.set_printoptions(precision = 10)
    del_J_bar_w1 = (1.0/number_of_training_data) * ( sum_del_Jp_w1 + ( lambdaval * w1) )

    #print("del_J_bar_w1:")
    #print(del_J_bar_w1.shape)
    #print(del_J_bar_w1)

    # compute equation (17):
    np.set_printoptions(precision = 10)
    del_J_bar_w2 = (1.0/number_of_training_data) * ( sum_del_Jp_w2 + ( lambdaval * w2) )

    #print("del_J_bar_w2:")
    #print(del_J_bar_w2.shape)
    #print(del_J_bar_w2)

	# compute equation (15):
    J_bar = J + (    (lambdaval / (2*26)) * (   np.sum(np.square(w1))   +   np.sum(np.square(w2))  )    )
    print("J_bar")
    print(J_bar)

    np.set_printoptions(precision = 10)
	#Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
	#you would use code similar to the one below to create a flat array
    obj_val = J_bar
    grad_w1 = del_J_bar_w1
    grad_w2 = del_J_bar_w2
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    #print("grad_w1")
    #print(grad_w1.shape)
    #print(grad_w1)

    #print("grad_w2")
    #print(grad_w2.shape)
    #print(grad_w2)

    #print("obj_grad")
    #print(obj_grad.shape)
    #print(obj_grad)

    pickleFile = open("params.pickle", 'wb')

    pickle.dump(n_hidden, pickleFile)
    pickle.dump(w1, pickleFile)
    pickle.dump(w2, pickleFile)
    pickle.dump(lambdaval, pickleFile)


    pickleFile.close()


    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    #################################################################################
    # added by: Zulkar
    # add bias 1 at (d+1) position of each data point
    number_of_training_data = data.shape[0]
    dimension_of_training_data = data.shape[1]

    # bias node is added to the training data:
    train_data_bias = np.ones( (number_of_training_data, dimension_of_training_data+1))
    train_data_bias[:,:-1] = data


    # computing dot product between data points and weights w1
    a_hidden = np.inner(train_data_bias,w1)
    #print("a_hidden: inner product between training data with bias and weight vector corresponds to hidden layer:")
    #print(a_hidden)

    # compute threshold function (sigmoid)
    z_hidden = sigmoid(a_hidden)
    #print("z_hidden: output of sigmoid function in hidden layer: ")
    #print(z_hidden)
    #print("z_hidden:shape")
    #print(z_hidden.shape)

    # add bias hidden node (m+1)th to z_hidden. We set its value 1 directly
    N1= z_hidden.shape[0]
    N2 = z_hidden.shape[1]
    z_hidden_bias = np.ones((N1,N2+1))
    z_hidden_bias[:,:-1] = z_hidden

    # computing dot product between hidden layer output z_hidden_bias and weights w2
    a_output = np.inner(z_hidden_bias,w2)
    #print("a_output: inner product between hidden layer output with bias and weight vector corresponds to output layer:")
    #print(a_output.shape)

    # compute threshold function (sigmoid)
    z_output = sigmoid(a_output)
    #print("z_output: output of sigmoid function in output layer: ")
    #print(z_output)
    #print("z_output:shape")
    #print(z_output.shape)


    # get the index of max element for each row
    labels = np.argmax(z_output, axis=1)
    #print("predicted label :")
    #print(labels)
    #print(labels.dtype)
    ######################################################################################
    #commented by : zulkar : 12:28am 2/29/16
    #labels = np.array([])
    #Your code here

    return labels

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 200;

# set the number of nodes in output unit
n_class = 10;
#print(n_class)

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 0.0;

# This part is not working right now ... need to update

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# This part is working


# To test this part we can send initial random weights to nnPredict(...) function
#################### this part is just for testing. Don't put it in final version ##########
#w1 = initial_w1
#w2 = initial_w2
#############################################################################################

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

printline1 = '\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%'
predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

printline2 = '\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%'
predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
#print(printline1)
#print(printline2)
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
