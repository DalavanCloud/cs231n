import numpy as np
from random import shuffle

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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Get Shapes
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    f_i = X[i,:].dot(W)
    #normalization trick to avoid numerical instability
    log_i = np.max(f_i)
    f_i -= log_i
    sum_i = np.sum(np.exp(f_i))
    loss += -f_i[y[i]] + np.log(sum_i)
    for j in xrange(num_classes):
      p = np.exp(f_i[j])/sum_i
      dW[:,j] += (p - (y[i] == j)) * X[i,:]

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #Get shape
  num_train = X.shape[0]
  num_classes = W.shape[1]

  score = X.dot(W)
  score = score - np.matrix(np.max(score, axis=1)).T
  score_correct = np.matrix(score[np.arange(num_train), y]).T
  loss = np.mean(-score_correct/np.log(np.sum(np.exp(score), axis=1)))

  p = np.exp(score)/np.sum(np.exp(score), axis=1)
  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = -1
  p += ind
  dW = X.T.dot(p)
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

