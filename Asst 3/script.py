import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
from sklearn.svm import SVC
from math import sqrt

#%matplotlib inline


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

	
def one_of_k(labels,k):
    # inputs : labels : the label vector that needs one of k encoding. dimension : N * 1 
    #          k : in our case k = 10
    
    N = labels.shape[0]

    # create an array of size N * k with all zeros
    result = np.zeros( (N , k) )
    
    # forcing labels to be integer:
    int_labels = labels.astype(int)
    
    row_index = 0
    for index in int_labels:
        result[row_index,index] = 1
        row_index = row_index + 1
    return result
	
	
	
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
	
	
	
def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    
    ################################# Start ######################################
    
    # added by : Zulkar : 4/18/16 2:23 pm 
    # add bias term at the beginning of the feature vector instead of the end. 
    train_data_with_bias = np.ones((n_data , n_features + 1))
    train_data_with_bias[:,1:] = train_data  # dim : N * D+1
    #print("train_data_with_bias:")
    #print(train_data_with_bias.shape)
    
    # compute theta_n = sigma(w.T,x_n)   
    # Since , initialWeights dim = (D+1) * 1
    #          train_data_with_bias dim = N * (D+1)
    # train_data_with_bias . initialWeights will give dim = N * 1
    
    W = initialWeights.reshape((n_feature+1,1))
    theta_n_temp = np.dot(train_data_with_bias,W)  # dim = N * 1
    theta_n = sigmoid(theta_n_temp)
    #print("theta_n:")
    #print (theta_n.shape)
    one_minus_theta_n = 1 - theta_n  # dim : N * 1
    
    ln_theta_n = np.log(theta_n)   # dim : N * 1
    
    ln_one_minus_theta_n = np.log(one_minus_theta_n)  # dim N * 1
    
    y_n = labeli   # dim : N * 1
    
    one_minus_y_n = 1 - labeli   # dim : N * 1
    
    yn_ln_thetan = y_n * ln_theta_n   # dim : N * 1
    
    one_minus_yn_thetan = one_minus_y_n * ln_one_minus_theta_n  # dim : N * 1
    
    add_both_part = yn_ln_thetan + one_minus_yn_thetan  # dim : N * 1
    
    e_w = np.sum(add_both_part)   # scalar
    error = (-1.0 / n_data) * e_w  # scalar
    
    #print (error)
    # added by : Zulkar : 4/18/16 2:23 pm
    ################################## end ###############################################
     
    # added by : Zulkar : 4/24/16 1:35 pm
    ################################## start ###############################################
    theta_n_minus_y_n = theta_n - y_n  # dim : N * 1
    
    # transpose the training data : 
    train_data_with_bias_transpose = np.transpose(train_data_with_bias)   # dim : (D+1) * N
    
    
    sum_theta_n_minus_y_n_into_xn = np.dot(train_data_with_bias_transpose, theta_n_minus_y_n)  #(D+1)*N . N*1
    
    error_grad_temp = (1.0 / n_data) * sum_theta_n_minus_y_n_into_xn
    error_grad = error_grad_temp.flatten()
    #print("error_grad:")
    #print(error_grad.shape)
    # added by : Zulkar : 4/24/16 1:35 pm
    ################################## end ###############################################
    

    return error, error_grad
	
	
	
def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # add bias term at the beginning of the feature vector instead of the end. 
    N = data.shape[0]
    D = data.shape[1]
    data_with_bias = np.ones(( N , D + 1))  # dim : N * (D+1)
    data_with_bias[:,1:] = data  # dim : N * D+1
    
    #data_with_bias_transpose = np.transpose(data_with_bias)  # dim : (D+1) * N
    wT_x = np.dot(data_with_bias , W) # dim :  (D+1) * N . (D + 1) x 10 = (D+1) *  10
    
    sigma_wT_x = sigmoid(wT_x)
    
    label_temp = np.argmax(sigma_wT_x, axis = 1)
    label = np.reshape(label_temp, (N,1) )
    
    return label

	
	
	
def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    W = params.reshape((n_feature+1,n_class))                                  #(716,10)

    # Formula 5 : Posterior_Probabilities1)                           #(50000,716)
    train_data_bias = np.insert(train_data, 0, 1, axis = 1)               #(50000,716)
    w_dot_x = np.dot(train_data_bias,W)          #wTx                    #(50000,10)
    exp_w_dot_x = np.exp(w_dot_x)                          #exp(wTx)               #(50000,10)
    sum_exp_w_dot_x = np.sum(exp_w_dot_x,axis = 1)      #sum(exp(wTx))          #(50000,1)
    inv_sum_exp_w_dot_x = 1.0 / sum_exp_w_dot_x

    posterior_probability = np.zeros((exp_w_dot_x.shape[0], exp_w_dot_x.shape[1]))

    for i in range(exp_w_dot_x.shape[0]):        #50000
        for k in range(exp_w_dot_x.shape[1]):
            posterior_probability[i][k] = exp_w_dot_x[i][k] * inv_sum_exp_w_dot_x[i]

    # Formula 6 : likelihood
    y_nk = one_of_k(posterior_probability,n_class)      #(50000,10)
    ln_theta_nk = np.log(posterior_probability)         #(50000,10)
    product_Y_nk_theta_nk = y_nk * ln_theta_nk
    sumK_product_Y_nk_theta_nk = np.sum(product_Y_nk_theta_nk, axis=1)    #(50000,1)
    sumN_sumK_product_Y_nk_theta_nk = np.sum(sumK_product_Y_nk_theta_nk, axis=0) #scalar
    error = (-1.0/n_data) * sumN_sumK_product_Y_nk_theta_nk                      #scalar
    print("error",error)

    # Formula 7 : log likelihood (error)

    # Formula 8 : gradient of error function

    difference = posterior_probability - y_nk;              #(50000,10)
    transpose_difference = np.transpose(difference)         #(10,50000)
    product = np.dot(transpose_difference,train_data_bias)  #(10,716) #scalar
    error_grad_temp2 = (1.0/n_data) * product        #(10,716)
    error_grad = error_grad_temp2.flatten()          #(7160,1)

    # Formula 9 : (not needed or used in the assignment)


    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad  
	
	
	
def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """

    label = np.zeros((data.shape[0], 1))
    data_bias = np.insert(data, 0, 1, axis = 1)    #(50000, 716)
    dot_product = np.dot(data_bias,W)                      #(50000, 10)
    label = np.argmax(dot_product, axis = 1)
    

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label
	
	
"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
"""
# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


with open('params.pickle', 'wb') as f1: 
    pickle.dump(W, f1)
"""

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################


# Linear kernel

"""
print('linear kernel')

clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())

train_acc = 100*clf.score(train_data, train_label)
print('\n Training Accuracy:' + str(train_acc) + '%')

test_acc = 100*clf.score(test_data, test_label)
print('\n Testing Accuracy:' + str(test_acc) + '%')

valid_acc = 100*clf.score(validation_data, validation_label)
print('\n Validation Accuracy:' + str(valid_acc) + '%')

"""


# Radial basis function:  gamma = 1


print('\n\n Radial basis function: gamma = 1')

clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())

train_acc = 100*clf.score(train_data, train_label)
print('\n Training Accuracy:' + str(train_acc) + '%')

test_acc = 100*clf.score(test_data, test_label)
print('\n Testing Accuracy:' + str(test_acc) + '%')

valid_acc = 100*clf.score(validation_data, validation_label)
print('\n Validation Accuracy:' + str(valid_acc) + '%')


# Radial basis function: gamma = 0
print('\n\n Radial basis function: gamma = 0')
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())


train_acc = 100*clf.score(train_data, train_label)
print('\n Training Accuracy:' + str(train_acc) + '%')

test_acc = 100*clf.score(test_data, test_label)
print('\n Testing Accuracy:' + str(test_acc) + '%')

valid_acc = 100*clf.score(validation_data, validation_label)
print('\n Validation Accuracy:' + str(valid_acc) + '%')




# Radial basis function with C being 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
print('\n\n SVM with  different values of C')
train_accuracy = np.zeros(11)
test_accuracy = np.zeros(11)
valid_accuracy = np.zeros(11)


C = 1.0
for i in range(11):
    clf = SVC(C=C, kernel='rbf')
    clf.fit(train_data, train_label.flatten())
    
    print ('\n C is :')
    print (C)
    
    train_accuracy[i] = 100*clf.score(train_data, train_label)
    print('\n Training  Accuracy for C : ' + str(train_accuracy[i]) + '%')
    
    test_accuracy[i] = 100*clf.score(test_data, test_label)
    print('\n Testing  Accuracy for C :'  + str(test_accuracy[i]) + '%')
    
    valid_accuracy[i] = 100*clf.score(validation_data, validation_label)
    print('\n Validation  Accuracy for C :'  + str(valid_accuracy[i]) + '%')
    
    if (i == 0):
        C = 10
    else:
        C = C + 10



# Plot accuracies
C_range = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plot(C_range, train_accuracy, 'o-',
    C_range, test_accuracy,'o-',
    C_range, valid_accuracy, 'o-')

ylabel('Accuracy (%)')
xlabel('Values of C')

title('Accuracy using SVM and different values of C')
legend(('Training','Test', 'Validation'), loc='lower right')
grid(True)
show()




"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
