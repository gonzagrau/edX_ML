import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
print('Loading data...')
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
# plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################
def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
# for lambda_factor in [1, 0.1, 0.001]:
#     print(f'Linear Regression test_error with {lambda_factor=} is: {run_linear_regression_on_MNIST(lambda_factor)}')
#

#######################################################################
# 3. Support Vector Machine
#######################################################################

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())

def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

def run_softmax_on_MNIST(temp_parameter: float=1.):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    return test_error


#print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

# temp_parameters = [.5, 1.0, 2.0]
# for temp in temp_parameters:
#     print(f'test_error with {temp=}', run_softmax_on_MNIST(temp_parameter=temp))

temp_parameter = 1.
#######################################################################
# 6. Changing Labels
#######################################################################

def run_softmax_on_MNIST_mod3(temp_parameter=1.):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y,
                                                      temp_parameter,
                                                      alpha=0.3,
                                                      lambda_factor=1.0e-4,
                                                      k=10,
                                                      num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    return test_error


# print(f"Test error mod 3 on current model: {run_softmax_on_MNIST(temp_parameter=1.)}")
# print(f"Test error mod 3 on new model: {run_softmax_on_MNIST_mod3(temp_parameter=1.)}")

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

def classify_with_PCA(n_components: int, plot: bool = False):
    ##Correction note:  the following 4 lines have been modified since release.
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.

    theta_pca, _ = softmax_regression(train_pca, train_y, temp_parameter,
                                                      alpha=0.3,
                                                      lambda_factor=1.0e-4,
                                                      k=10,
                                                      num_iterations=150)

    test_err_pca = compute_test_error(test_pca, test_y, theta_pca, temp_parameter=1.)
    print(f"PCA softmax test error with {n_components=}: {test_err_pca}")

    if plot:
        plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means) #feature_means added since release

        firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
        plot_images(firstimage_reconstructed)
        plot_images(train_x[0, ])

        secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
        plot_images(secondimage_reconstructed)
        plot_images(train_x[1, ])

        return train_x_centered, feature_means, train_pca, test_pca

# classify_with_PCA(10, True)

## Cubic Kernel ##
print('Getting PCA sets...')
n_components = 10
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# print('Extracting cubic features...')
# train_cubic_pca = cubic_features(train_pca)
# test_cubic_pca = cubic_features(test_pca)
# print('Training the model...')
# theta_pca_cubic, _ = softmax_regression(train_cubic_pca, train_y, temp_parameter,
#                                           alpha=0.3,
#                                           lambda_factor=1.0e-4,
#                                           k=10,
#                                           num_iterations=150)
# print('Computing error...')
# test_err_cubic = compute_test_error(test_cubic_pca, test_y, theta_pca_cubic, temp_parameter=1.)
# print(f"PCA softmax test error with {n_components=}: {test_err_cubic}")
print('Fitting cubic SVM...')
cubic_SVM = SVC(kernel="poly", degree=3, random_state=0)
cubic_SVM.fit(train_pca, train_y)

y_pred_cub = cubic_SVM.predict(test_pca)
print(f'Test accuracy: {1 - accuracy_score(test_y, y_pred_cub):.4f}')

print('Fitting gaussian SVM...')
gauss_SVM = SVC(kernel="rbf", random_state=0)
gauss_SVM.fit(train_pca, train_y)

y_pred_gauss = gauss_SVM.predict(test_pca)
print(f'Test accuracy: {1 - accuracy_score(test_y, y_pred_gauss):.4f}')

