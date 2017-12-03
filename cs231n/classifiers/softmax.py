import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
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

  for i in xrange(0,y.shape[0]):
    loss-=f[i][y[i]]
    for j in xrange(0,X.shape[1]):
      loss += np.log(np.exp(f[i][j]))

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
  f = np.exp(f)
  loss_log_bottom =np.log( np.sum(f,axis = 1) )


  diff = np.square(f - np.transpose(y))
  y_i_idx = np.argmin(diff,axis=1)
  loss_log_top = np.log( f[np.arange(num_train),y_i_idx] )

  loss = np.sum( -loss_log_top + loss_log_bottom + np.sum(np.square(W),axis = 1))
  loss /= num_train



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

