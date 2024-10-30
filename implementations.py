import numpy as np
from helpers import *

################## mean_sqared_error_gd
def compute_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum

    >>> compute_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y-np.dot(tx,w)
    grad = -1/len(tx) *np.dot(np.transpose(tx),e)
    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: last model parameters
        loss: last loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w-gamma * gradient
    loss = compute_mse(y, tx, w)
    return w,loss
    
################## mean_squared_error_sgd

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N = len(y)
    txT = np.reshape(tx,(2,-1))
    e = np.reshape(y-np.reshape(np.dot(tx,w),(1,-1)),(-1,1))
    grad = -1/N * (np.dot(txT,e))
    grad = np.reshape(grad, ((1,-1)))
    return grad


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: last model parameters
        loss: last loss value
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma * gradient
    loss = compute_mse(y, tx, w)
    return w,loss
    
################## least_squares
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # returns mse, and optimal weights
    wstar = np.linalg.solve(np.dot(np.transpose(tx), tx), np.dot(np.transpose(tx), y))
    mse = compute_mse(y, tx, wstar)
    return wstar, mse

################## ridge_regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = np.shape(tx)[0]
    D = np.shape(tx)[1]
    lambda_prime = lambda_ * 2 * N
    I = np.eye(D)
    A = np.dot(np.transpose(tx), tx) + np.dot(lambda_prime, I)
    B = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(A, B)
    loss = compute_mse(y, tx, w)
    return w,loss

################## logistic_regression
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    #raise NotImplementedError
    sigma = (1+np.exp(-t))**(-1)
    return (sigma)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    N = len(y)
    t = np.dot(tx,w) # N*1
    L= (-1/N) * (y*np.log(sigmoid(t))+(1-y)*np.log(1-sigmoid(t))).sum() # * for element-wise
    return(L)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    N = np.shape(y)[0]
    t = np.dot(tx,w)
    grad = (1/N)*np.dot(np.transpose(tx),sigmoid(t)-y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    w_new = w-gamma*calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    return w_new,loss

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    N = np.shape(y)[0]
    t = np.dot(tx,w) 
    diag_elements = sigmoid(t)*(1-sigmoid(t))
    S = np.diag(diag_elements.flatten())
    hessian = (1/N)*np.dot(np.dot(np.transpose(tx),S),tx) # (3 2)(2 2)(2 3)
    return hessian

def loss_grad_hess(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """
    Loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return Loss,gradient,hessian
    
def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """Performs logistic regression using gradient descent.

    Args:
        y: array of shape (N, 1) - Target values, where N is the number of samples.
        tx: array of shape (N, D) - Feature matrix, where D is the number of features.
        initial_w: array of shape (D, 1) - Initial weights for the regression.
        max_iters: int - Maximum number of iterations to perform in the gradient descent.
        gamma: float - Step size or learning rate for the gradient descent.

    Returns:
        w: array of shape (D, 1) - Optimized weights after convergence or reaching max iterations.

    Example:
        >>> y = np.array([[1], [0]])
        >>> tx = np.array([[1, 2], [3, 4]])
        >>> initial_w = np.array([[0.5], [0.5]])
        >>> max_iters = 1000
        >>> gamma = 0.1
        >>> w = logistic_regression(y, tx, initial_w, max_iters, gamma)
        Current iteration=0, loss=0.6931471805599453
        Current iteration=100, loss=0.5931471805599453
        ...
    """
    threshold = 1e-10
    losses = []
    w = initial_w
    
    if max_iters == 0:
        return w,calculate_loss(y, tx, w)
    else: 
        # start the logistic regression
        for iter in range(max_iters):
            # get loss and update w.
            w,loss = learning_by_gradient_descent(y, tx, w, gamma)
            #print(w.shape)
            # log info
            #if iter % 100 == 0:
                #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
                
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break 
        return w,losses[-1]
'''
def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 0., 1., 1.]]
    >>> np.random.seed(0)
    >>> tx = np.random.rand(4, 3)
    >>> w = np.array([[0.1], [0.5], [0.5]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_newton_method(y, tx, w, gamma)
    >>> round(loss, 8)
    0.71692036
    >>> w
    array([[-1.31876014],
           [ 1.0590277 ],
           [ 0.80091466]])
    """

    loss,gradient,hessian = logistic_regression(y, tx, w)
    w = w-gamma*np.dot(np.linalg.inv(hessian),gradient)
    return w,loss
'''
def penalized_logistic_regression(y, tx, w,lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar
        max_iters:
        gamma:

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    loss, gradient, _ = loss_grad_hess(y, tx, w)
    penalized_loss = loss + lambda_*np.linalg.norm(w)**2 
    penalized_gradient = gradient + 2*lambda_*w

    return loss, penalized_gradient

def learning_by_penalized_gradient(y,tx,w,gamma,lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        max_iters:
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    loss, gradient = penalized_logistic_regression(y, tx, w,lambda_)
    w = w-gamma*gradient
    return w,loss

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """Performs regularized logistic regression using penalized gradient descent.

    Args:
        y: array of shape (N, 1) - Target values, where N is the number of samples.
        tx: array of shape (N, D) - Feature matrix, where D is the number of features.
        lambda_: float - Regularization parameter to control overfitting.
        initial_w: array of shape (D, 1) - Initial weights for the regression.
        max_iters: int - Maximum number of iterations for the gradient descent.
        gamma: float - Step size or learning rate for the gradient descent.

    Returns:
        w: array of shape (D, 1) - Optimized weights after convergence or reaching max iterations.

    Example:
        >>> y = np.array([[1], [0]])
        >>> tx = np.array([[1, 2], [3, 4]])
        >>> lambda_ = 0.1
        >>> initial_w = np.array([[0.5], [0.5]])
        >>> max_iters = 1000
        >>> gamma = 0.1
        >>> w = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
        Current iteration=0, loss=0.6931471805599453
        Current iteration=100, loss=0.5931471805599453
        ...
        loss=0.5001471805599453
    """
    threshold = 1e-10
    losses = []
    w = initial_w
    if max_iters == 0:
        return w,calculate_loss(y, tx, w)
    else: 
        # start the logistic regression
        for iter in range(max_iters):
            # get loss and update w.
            w,loss = learning_by_penalized_gradient(y,tx,w,gamma,lambda_)
            # log info
            #if iter % 100 == 0:
                #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
        #print("loss={l}".format(l=compute_mse(y, tx, w)))
        return w,losses[-1]
    

