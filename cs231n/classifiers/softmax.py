import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C)
  containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  f = np.matmul(X,W)
  p = np.zeros_like(f)


  for i in xrange(0,y.shape[0]):
    f[i] -= np.max(f[i]) #normalize
    np.exp(f[i])
    f_y_i = f[i][y[i]]
    f_sum = 0
    for j in xrange(0,X.shape[1]):
      f_sum += f[i][j]
    p[i] = f_y_i / f_sum
    l = np.log(p[i])
    dl_do = 0.3 * p[i] *(-1)
    loss -= l
    dW[:,y[i]] += (dl_do - np.square(dl_do)) * np.transpose(X[i,:]) 


  loss /= num_train


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape(0)
  f = np.matmul(X,W)
  f -=  np.max(f)
  p = np.exp(f) / np.sum(np.exp(f))
  f = np.exp(f)


  loss_list = p[np.arange(num_train),y]


  loss = np.sum(loss_list)
  loss /= num_train



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

