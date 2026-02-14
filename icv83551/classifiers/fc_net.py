from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        # First we make a flat list of all the layers/sizes in the modular network
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        # Loop over the layers and build them
        for i in range(self.num_layers):
            W_name = 'W%d' % (i+1)
            b_name = 'b%d' % (i+1)
            
            dim_in = layer_dims[i]
            dim_out = layer_dims[i+1]
            
            # Initialize weights like was asked
            self.params[W_name] = np.random.randn(dim_in, dim_out) * weight_scale
            self.params[b_name] = np.zeros(dim_out)
            
            # Initialize batch norm params like was asked
            if self.normalization in ['batchnorm', 'layernorm'] and i < self.num_layers - 1:
                gamma_name = 'gamma%d' % (i + 1)
                beta_name = 'beta%d' % (i + 1)
                
                self.params[gamma_name] = np.ones(dim_out)
                self.params[beta_name] = np.zeros(dim_out)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        out = X
        caches = []
        
        # Loop over all layers except last ones
        for i in range(self.num_layers - 1):
            W = self.params['W%d' % (i + 1)]
            b = self.params['b%d' % (i + 1)]
            
            out, cache_affine = affine_forward(out, W, b)
            
            # Cache norms
            cache_norm = None 
            if self.normalization == 'batchnorm':
                gamma = self.params['gamma%d' % (i + 1)]
                beta = self.params['beta%d' % (i + 1)]
                bn_param = self.bn_params[i]
                out, cache_norm = batchnorm_forward(out, gamma, beta, bn_param)
            elif self.normalization == 'layernorm':
                gamma = self.params['gamma%d' % (i + 1)]
                beta = self.params['beta%d' % (i + 1)]
                ln_param = self.bn_params[i] 
                out, cache_norm = layernorm_forward(out, gamma, beta, ln_param)    
            
            out, cache_relu = relu_forward(out) # ReLU cache
            
            cache_dropout = None
            if self.use_dropout:
                # Dropout layer after ReLU
                out, cache_dropout = dropout_forward(out, self.dropout_param)
            
            # Store everything
            if self.normalization in ['batchnorm', 'layernorm']:
                caches.append((cache_affine, cache_norm, cache_relu, cache_dropout))
            else:
                caches.append((cache_affine, cache_relu, cache_dropout))
        
        # Output layers
        W = self.params['W%d' % self.num_layers]
        b = self.params['b%d' % self.num_layers]
        
        scores, cache_affine = affine_forward(out, W, b)
        caches.append(cache_affine)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y) # Loss and gradient
        
        # Output layer backprop
        last_layer_idx = self.num_layers
        last_cache = caches[last_layer_idx - 1]
        
        dout, dW, db = affine_backward(dscores, last_cache)
        
        # Regularization gradients
        grads['W%d' % last_layer_idx] = dW + self.reg * self.params['W%d' % last_layer_idx]
        grads['b%d' % last_layer_idx] = db
        
        # Sum loss
        loss += 0.5 * self.reg * np.sum(self.params['W%d' % last_layer_idx]**2)
        
        # Backprop all
        for i in range(self.num_layers - 2, -1, -1):
            layer_idx = i + 1
            
            # Unpack cache with and without norms
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                affine_cache, bn_cache, relu_cache, dropout_cache = caches[i]
            else:
                affine_cache, relu_cache, dropout_cache = caches[i]
                
            if self.use_dropout:
                # Dropout backward before ReLU backward
                dout = dropout_backward(dout, dropout_cache)    
            
            dout = relu_backward(dout, relu_cache) # ReLU

            # Batch Norm Backward
            if self.normalization == 'batchnorm':
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads['gamma%d' % layer_idx] = dgamma
                grads['beta%d' % layer_idx] = dbeta
                
            elif self.normalization == 'layernorm':
                dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
                grads['gamma%d' % layer_idx] = dgamma
                grads['beta%d' % layer_idx] = dbeta    

            dout, dW, db = affine_backward(dout, affine_cache) # Affine
            
            # Store gradients
            grads['W%d' % layer_idx] = dW + self.reg * self.params['W%d' % layer_idx]
            grads['b%d' % layer_idx] = db
            
            # Sum all the loss like before
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % layer_idx]**2)
         
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
