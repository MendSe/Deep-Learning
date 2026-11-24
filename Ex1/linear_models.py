"""
Linear Models skeleton for Deep Learning HW1.

This module defines the classes and functions that students need to complete.
The notebook imports these definitions instead of defining them inline.

Fill in every block marked TODO / START OF YOUR CODE.
"""
import numpy as np
from typing import Dict, Tuple, Iterable, Optional, Any


class LinearClassifier:
    """
    Base class for linear classifiers.  Stores weights and provides the
    interface for prediction, training, and computing accuracy.  Subclasses
    should override ``predict`` and ``loss`` to implement specific
    classification algorithms.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the classifier with a small random weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Training data matrix of shape (N, D), where N is the number
            of samples and D is the number of features (possibly including
            a bias term).  The weight matrix will have shape (D, C).

        y : np.ndarray
            Array of shape (N,) containing integer class labels in the
            range [0, C-1], where C is the number of distinct classes.
        """
        N, D = X.shape
        C = int(np.max(y)) + 1
        # Initialize weights with small random values
        self.W = 0.001 * np.random.randn(D, C)
        self.num_classes = C
        self.num_features = D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given data using the classifier's weights.

        This default implementation should be overridden by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (M, D) of input data.

        Returns
        -------
        np.ndarray
            Array of shape (M,) of predicted class labels.
        """
        raise NotImplementedError("predict method must be implemented in subclass")
        #predictions = X @ self.W



    def calc_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on a dataset.

        Accuracy is the fraction of instances that are classified correctly.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        y : np.ndarray
            True labels of shape (M,).

        Returns
        -------
        float
            Accuracy as a float in the range [0, 1].
        """
        accuracy = 0.0
        ###########################################################################
        # TODO: Implement this method.                                            #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################
        correctly_classified = np.sum(self.predict(X) == y)
        accuracy = correctly_classified / y.shape[0]
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return accuracy

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              learning_rate: float = 1e-3,
              num_iters: int = 100,
              batch_size: int = 200,
              verbose: bool = False) -> list:
        """
        Train the classifier using stochastic gradient descent.

        This method samples minibatches, computes the loss and gradient,
        and performs weight updates.  It collects the loss value at each
        iteration in ``loss_history`` and returns it.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        learning_rate : float, optional
            Step size for gradient descent (default: 1e-3).

        num_iters : int, optional
            Number of iterations to run (default: 100).

        batch_size : int, optional
            Number of samples per minibatch (default: 200).

        verbose : bool, optional
            If True, prints loss every 100 iterations.

        Returns
        -------
        list
            A list containing the loss value at each iteration.
        """
        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in every iteration.                       #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch                                                               #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #                                                                       #
        # Next, calculate the loss and gradient and update the weights using    #
        # the learning rate. Use the loss_history array to save the loss on     #
        # iteration to visualize the loss.                                      #
        #########################################################################
        num_instances = X.shape[0]
        loss_history = []
        loss = 0.0
        for i in range(num_iters):
            X_batch = None
            y_batch = None
            ###########################################################################
            # TODO: Create X_batch and y_batch. Call the loss method to get the loss value  #
            # and grad (the loss method is overridden in each subclass, see its       #
            # return values).                                                         #
            # Finally, append each of the loss values created in each iteration       #
            # to loss_history.                                                        #
            ###########################################################################
            #                          START OF YOUR CODE                             #
            ###########################################################################
            indices = np.random.choice(X.shape[0],replace=True,size=batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            batch_loss = self.loss(X_batch,y_batch)
            loss, grad = batch_loss[0], batch_loss[1]
            loss_history.append(loss)
            ###########################################################################
            #                           END OF YOUR CODE                              #
            ###########################################################################
            # TODO:                                                                   #
            # Perform parameter update                                                #
            # Update the weights using the gradient and the learning rate.            #
            ###########################################################################
            #                          START OF YOUR CODE                             #
            ###########################################################################
            self.W -= learning_rate*grad

            ###########################################################################
            #                       END OF YOUR CODE                                  #
            ###########################################################################

            if verbose and i % 100 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def loss(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the loss function and its gradient.

        Subclasses should override this method to compute the appropriate
        loss and gradient for their algorithm.

        Parameters
        ----------
        X : np.ndarray
            Minibatch of data of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` is
            an array of the same shape as ``self.W``.
        """
        raise NotImplementedError("loss must be implemented in subclass")


class LinearPerceptron(LinearClassifier):
    """
    Linear classifier that uses the Perceptron loss.
    Students should implement ``__init__`` and ``predict``; ``loss`` is provided.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the perceptron using the base class constructor.
        """
        ###########################################################################
        # TODO: Initiate the parameters of your model.                            #
        # You can assume y takes values 0...K-1 where K is number of classes      #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################
        LinearClassifier.__init__(self,X,y)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the perceptron rule.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        y_pred = None
        ###########################################################################
        # TODO: Implement this method.                                            #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################
        z = X @ self.W
        y_pred = np.argmax(z, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute perceptron loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        # Do not change this function. The loss itself is implemented in `perceptron_loss_naive`.
        return perceptron_loss_naive(self.W, X_batch, y_batch)


class LogisticRegression(LinearClassifier):
    """
    Linear classifier that uses softmax and cross-entropy loss for multiclass
    classification.
    Students should implement ``__init__`` and ``predict``; ``loss`` is provided.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the logistic regression model using the base class constructor.
        """
        self.W = None
        ###########################################################################
        # TODO: Initialize the model via the base class constructor.              #
        ###########################################################################
        super().__init__(X,y)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the softmax probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        y_pred = None
        ###########################################################################
        # TODO: Implement this method.                                                  #
        ###########################################################################
        z = X @ self.W
        probs = softmax(z)
        y_pred = np.argmax(probs, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute softmax cross-entropy loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        # Do not change this function. The loss itself is implemented in
        # `softmax_cross_entropy_vectorized`.
        return softmax_cross_entropy_vectorized(self.W, X_batch, y_batch)

def perceptron_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass perceptron loss using explicit loops.

    For each sample i, with scores s_i = x_i W and predicted class
    yhat_i = argmax_j s_ij, the per-sample loss is

        L_i = max_j s_ij - s_{i, y_i}      (>= 0, and 0 when the prediction is correct)

    For a misclassified sample the gradient contribution is
        dW[:, yhat_i] += x_i
        dW[:, y_i]    -= x_i
    and 0 for a correctly classified sample.

    The returned loss and gradient are both averaged over N. No regularization.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
        Average perceptron loss over the batch
    dW : (D, C)
        Gradient of the loss w.r.t. W
    """
    assert X.shape[1] == W.shape[0], \
        f"X has {X.shape[1]} features but W expects {W.shape[0]} (check the bias column)"
    X_use = X

    N = X_use.shape[0]
    _, C = W.shape

    # Initialize loss & gradient
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Implement Perceptron loss with explicit loops                     #
    #                                                                         #
    # After looping over all samples:                                         #
    #   - Average loss and gradient by N                                      #
    #############################################################################
    labels = np.unique(y)
    for i in range(len(y)):
        scores = W.T @ X_use[i]
        y_hat = 0
        max_score = scores[0]
        for j in range(1, C):
            if scores[j]>max_score:
                max_score = scores[j]
                y_hat = j

        loss += max_score - scores[y[i]]

        if y_hat != y[i]:
            dW[:, y_hat] += X_use[i]
            dW[:, y[i]] -= X_use[i]

    loss /= N
    dW /= N
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW

def softmax_cross_entropy(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient
    using explicit loops over the samples (naive version).

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    assert X.shape[1] == W.shape[0], \
        f"X has {X.shape[1]} features but W expects {W.shape[0]} (check the bias column)"
    X_use = X

    N = X_use.shape[0]
    loss, dW = 0.0, np.zeros_like(W)

    #############################################################################
    # TODO: Implement the forward pass with explicit loops.                     #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################
    C = W.shape[1]
    scores = np.zeros((N,C))
    probs = np.zeros((N,C))
    log_denoms = np.zeros(N)

    for i in range(N):
        scores[i] = X_use[i] @ W
        scores[i] -= np.max(scores[i])

        exp_scores = np.exp(scores[i])
        sum_exp  = np.sum(exp_scores)

        probs[i] = exp_scores / sum_exp
        log_denoms[i] = np.log(sum_exp)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


    #############################################################################
    # TODO: Compute the loss.                                                   #
    # Use the average negative log-likelihood of the correct class.             #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################
    for i in range(N):
        loss -= scores[i, y[i]] - log_denoms[i]
    loss /= N
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


    #############################################################################
    # TODO: Backward pass: compute gradient dW.                                 #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################
    for i in range(N):
        for j in range(C):
            dW[:,j] += (probs[i,j] - (j==y[i])) * X_use[i]

    dW /= N
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return loss, dW



def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function for each row of the input x.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N, C) where N is the number of samples and C
        is the number of classes.

    Returns
    -------
    probs : (N, C)
        Row-wise probabilities that sum to 1
    """
    probs = np.zeros_like(x)
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################
    N, C = x.shape
    probs = np.zeros((N,C))
    for i in range(N):
        exps = np.exp(x[i] - np.max(x[i]))
        normalizing = np.sum(exps)
        
        for j in range(C):
            probs[i,j] = exps[j] / normalizing
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return probs

def softmax_cross_entropy_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient
    using a fully vectorized implementation.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    assert X.shape[1] == W.shape[0], \
        f"X has {X.shape[1]} features but W expects {W.shape[0]} (check the bias column)"
    X_use = X

    N = X_use.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Implement the forward pass in a fully vectorized way.               #
    # 1. Compute the scores (N, C) for all samples in X_use.                    #
    # 2. No extra stabilizing step: softmax() already subtracts each row's max. #
    # 3. Compute the probabilities (N, C) with your softmax() function.         #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    C = W.shape[1]
    scores = X_use @ W
    probs = softmax(scores)
    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Compute the loss.                                                   #
    # 1. Select the probabilities for the correct class for all samples.        #
    #    (Hint: Use advanced integer indexing with np.arange(N) and y)          #
    # 2. Compute the negative log-likelihood for these probabilities.           #
    # 3. Compute the average loss (a scalar) across all samples in the batch.   #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    correct_probs = probs[np.arange(N), y]
    loss -= np.mean(np.log(correct_probs))

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Backward pass: compute gradient dW.                                 #
    # 1. Compute the gradient of the loss with respect to the scores.           #
    # 2. Compute the gradient dW using the chain rule (X_use.T @ dscores).      #
    # 3. Average the gradient over the batch (divide by N).                     #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1

    dW = X_use.T @ dscores
    dW /= N

    

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################

    return loss, dW


def tune_hyperparameters(
    ModelClass,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
    *,
    num_iters: int = 500,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[float, int], Tuple[float, float]], Any, float]:
    """
    Hyperparameter sweep for any LinearClassifier subclass
    (LinearPerceptron or LogisticRegression).

    Parameters
    ----------
    ModelClass : class
        Class with API:
          - __init__(X_init, y_init, **kwargs)   # optional; ok if ignores data
          - train(X, y, learning_rate, num_iters, batch_size, verbose=False) -> loss_history
          - calc_accuracy(X, y) -> float
    X_train, y_train, X_val, y_val : arrays
        Training/validation data.
    learning_rates : iterable of float
    batch_sizes : iterable of int
    num_iters : int
        Iterations per configuration.
    model_kwargs : dict or None
        Extra kwargs to pass to ModelClass.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    results : dict
        {(lr, batch): (train_acc, val_acc)} for each tried combo.
    best_model : object
        The fitted model instance with the highest validation accuracy.
    best_val : float
        The best validation accuracy achieved.
    """

    # Initialization
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    results: Dict[Tuple[float, int], Tuple[float, float]] = {}
    best_val = -1.0
    best_model = None

    ############################################################################
    # TODO: Iterate over all combinations of learning_rates and batch_sizes.   #
    #   For each (lr, batch_size):                                             #
    #     1. Create a new model using ModelClass.                              #
    #     2. Train it using the train() method with the given lr and batch.    #
    #     3. Compute training and validation accuracies using calc_accuracy(). #
    #     4. Store results[(lr, batch)] = (train_acc, val_acc).                #
    #     5. Track the best model based on validation accuracy.                #
    #                                                                          #
    # Hints:                                                                   #
    # - Use 'verbose' to optionally print current hyperparameters.             #
    # - Make sure to create a *new* model for each configuration.              #
    ############################################################################
    for lr in learning_rates:
        for batch in batch_sizes:
            # TODO: Create a new model instance
            model = ModelClass(X_train, y_train, **model_kwargs)

            # TODO: Train the model using the provided hyperparameters
            model.train(X_train, 
                        y_train,
                        learning_rate=lr,
                        num_iters=num_iters,
                        batch_size=batch,
                        verbose=verbose)

            # TODO: Compute train and validation accuracy
            train_acc = model.calc_accuracy(X_train,y_train)
            val_acc = model.calc_accuracy(X_val,y_val)

            # TODO: Store the results in the 'results' dictionary
            results[(lr, batch)] = (train_acc, val_acc)

            # TODO: Update the best model and best_val if this is the best so far
            if val_acc > best_val :
                best_val = val_acc
                best_model = model

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return results, best_model, best_val

