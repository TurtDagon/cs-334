"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (N, M+1)
    """
    # TODO: Implement this function
    Phi = np.zeros((X.shape[0],M+1))
    for idx, x in enumerate(X):
        for m in range(M):
            Phi[idx, m] = x[0]**m
        Phi[idx, m+1] = x[0]**(m+1)            
    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)
    
    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    # TODO: Implement this function
    sum = 0
    for idx,i in enumerate(y):
        tmp = i - np.dot(theta,X[idx])
        sum += (tmp**2)/2
    loss = sum/X.shape[0]
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    #E_rms =  
    sum = 0
    for i in range(X.shape[0]):
        sum += (y[i] - np.dot(theta,X[i]))**2
    tmp = sum/X.shape[0]
    E_rms = np.sqrt(tmp)
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    theta =  np.zeros(X.shape[1])
    prev_loss = np.mean((y - X @ theta) ** 2) / 2
    i = 0
    while i <=1e6:
        # and i < y.shape[0]
        grad = np.mean((y - X @ theta)[:, None] * X, axis=0)
        theta =  theta + learning_rate * grad
        i+=1 
        new_loss = np.mean((y - X @ theta) ** 2) / 2
        if abs(new_loss - prev_loss) <= 1e-10:
            break
        prev_loss = new_loss

    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N = X.shape[0]
    theta = np.zeros(X.shape[1])
    prev_loss = np.mean((y - X @ theta) ** 2) / 2
    i = 0
    while i < 1e6:
        tmp = 1 / (0.001 * i+1) if learning_rate == 'adaptive' else learning_rate
        j = i % N
        theta = theta + tmp * (y[j] - np.dot(theta, X[j])) * X[j]
        i += 1

        if i % N == 0:  #stop full pass over data
            new_loss = np.mean((y - X @ theta) ** 2) / 2
            if abs(new_loss - prev_loss) <= 1e-10:
                break
            prev_loss = new_loss
    return theta

def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d) #20,2
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    # theta = (XTX)-1XT * Y
    #L2: theta = inv(XTX+lambda)XTy
    theta = np.linalg.inv(X.T @ X) @ X.T @ y 
    L2theta = np.linalg.inv(X.T @ X + reg_param * np.eye(X.shape[1])) @ X.T @ y
    return L2theta


#Uncomment this if you are attempting the extra credit
def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    W = np.diag(weights)
    tmp = np.linalg.inv(X.T @ W @ X + reg_param * np.eye(X.shape[1]))
    theta = tmp @ X.T @ W @ y
    return theta



def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    # Example of how to use the functions
    # start = time.process_time()
    # theta, i = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=0.1)
    # print('Time elapsed:', time.process_time() - start)
    
    # TODO: Add more code here to complete part 1
    ##############################
    # start = time.process_time()
    # theta = ls_stochastic_gradient_descent(Phi_train, y_train)
    # print('Time elapsed:', time.process_time() - start)
    # print(theta)

    # print("SGD!!!")
    # for lr in (0.1,0.01,0.001,0.0001):
    #     print(lr)
    #     start = time.process_time()
    #     theta, i = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=lr)
    #     print('Time elapsed:', time.process_time() - start)
    #     print(theta)
    #     print(i)
    # print("GD")

    # for lr in (0.1,0.01,0.001,0.0001):
    #     print(lr)
    #     start = time.process_time()
    #     #remove the if you don't need the iteration number
    #     theta, i = ls_gradient_descent(Phi_train, y_train, learning_rate=lr)
    #     print('Time elapsed:', time.process_time() - start)
    #     print(theta)
    #     print(i)
        
    # start = time.process_time()
    # theta = ls_closed_form_solution(Phi_train,y_train) 
    # print('Time elapsed:', time.process_time() - start)
    # print(theta)
    

def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # TODO: Add more code here to complete part 2
    ##############################
    y1 = []
    y2 = []
    x = range(1,11)
    # for M in x:
    #     Phi_train = generate_polynomial_features(X_train, M)
    #     Phi_val = generate_polynomial_features(X_validation,M)
    #     theta = ls_closed_form_solution(Phi_train,y_train)

    #     y1.append(calculate_RMS_Error(Phi_train, y_train, theta))
    #     y2.append(calculate_RMS_Error(Phi_val, y_validation, theta))

    
    #np.linspace(10**-8,10**0, 9) ? i dunno
    regs = [0,10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,0.1, 1]
    x = np.arange(len(regs))
    Phi_train = generate_polynomial_features(X_train, 10)
    Phi_val = generate_polynomial_features(X_validation,10)
    for rp in regs:
        theta = ls_closed_form_solution(Phi_train,y_train, rp)

        y1.append(calculate_RMS_Error(Phi_train, y_train, theta))
        y2.append(calculate_RMS_Error(Phi_val, y_validation, theta))
    labels = ["0", "$10^{-8}$", "$10^{-7}$", "$10^{-6}$", "$10^{-5}$","$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"]

    plt.xticks(x, labels, rotation = 0)
    plt.title('L2 Regularization of Training vs. Validation rms Error')
    plt.plot(x, y1, 'ro-', label="Training", linewidth = 1)
    plt.plot(x, y2, 'bo-', label="Validation", linewidth=1)
    plt.xlabel('Regularization λ')
    plt.ylabel('E_rms')
    plt.legend() 
    plt.tight_layout()
    plt.show()

    print("Done!")


#Uncomment this if you are attempting the extra credit
def extra_credit(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete extra credit
    """
    print("=========== Extra Credit ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete the extra credit
    ##############################
    y1 = []
    y2 = []
    x = range(1,11)
    for M in x:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_val = generate_polynomial_features(X_validation,M)
        theta = weighted_ls_closed_form_solution(Phi_train,y_train,weights_train)

        y1.append(calculate_RMS_Error(Phi_train, y_train, theta))
        y2.append(calculate_RMS_Error(Phi_val, y_validation, theta))

    plt.title('Weighted LR of Training vs. Validation rms Error')
    plt.plot(x, y1, 'ro-', label="Training", linewidth = 1)
    plt.plot(x, y2, 'bo-', label="Validation", linewidth=1)
    plt.xlabel('M')
    plt.ylabel('E_rms')
    plt.legend() 
    plt.tight_layout()
    plt.show()
    print("Done!")


def main(fname_train, fname_validation):
    #part_1(fname_train)
    #part_2(fname_train, fname_validation)
    extra_credit(fname_train, fname_validation)


if __name__ == '__main__':
    main("data/linreg_train.csv", "data/linreg_validation.csv")
