from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad) / self.input.shape[0]
        self.grads['b'] = np.sum(grad, axis=0) / self.input.shape[0]
        return np.dot(grad, self.W.T)
        
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:    
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W': None, 'b': None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.input = None

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        """
        self.input = X
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        else:
            X_padded = X
        self.input_padded = X_padded 

        batch_size, in_channels, H, W = X_padded.shape
        out_channels, _, kH, kW = self.W.shape

        new_H = (H - kH) // self.stride + 1
        new_W = (W - kW) // self.stride + 1

        out = np.zeros((batch_size, out_channels, new_H, new_W))

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = X_padded[b, :, h_start:h_start+kH, w_start:w_start+kW]
                        out[b, oc, i, j] = np.sum(window * self.W[oc]) + self.b[oc]
        return out

        

    def backward(self, grads):
        """
        grads: [batch_size, out_channel, new_H, new_W]
        """
        batch_size, in_channels, H, W = self.input.shape
        out_channels, _, kH, kW = self.W.shape
        _, _, new_H, new_W = grads.shape

        dX_padded = np.zeros_like(self.input_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = self.input_padded[b, :, h_start:h_start+kH, w_start:w_start+kW]

                        dW[oc] += grads[b, oc, i, j] * window
                        dX_padded[b, :, h_start:h_start+kH, w_start:w_start+kW] += grads[b, oc, i, j] * self.W[oc]
                        db[oc] += grads[b, oc, i, j]

        # 去掉 padding 
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        if self.input.shape == grads.shape:
            output = np.where(self.input < 0, 0, grads)
            return output
        else:
            assert np.prod(self.input.shape) == np.prod(grads.shape), f'The shape of input and grads are not the same.: {self.input.shape} v.s. {grads.shape}'
            output = np.where(self.input < 0, 0, grads.reshape(self.input.shape))
            return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.softmax_output = None
        self.preds = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.preds = predicts
        self.labels = labels
        if self.has_softmax:
            self.softmax_output = softmax(predicts)
        else:
            self.softmax_output = predicts
        
        batch_size = predicts.shape[0]
        correct_probs = self.softmax_output[np.arange(batch_size), labels]
        loss = -np.mean(np.log(correct_probs + 1e-12)) 
        return loss
    
    def backward(self):
        batch_size = self.labels.shape[0]
        grads = self.softmax_output.copy()
        grads[np.arange(batch_size), self.labels] -= 1
        grads /= batch_size
        self.grads = grads
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self):
        l2_loss = 0
        for param_name in self.model.grads:
            param = getattr(self.model, param_name)
            l2_loss += np.sum(param ** 2)
        return self.weight_decay_lambda * 0.5 * l2_loss

    def backward(self):
        for param_name in self.model.grads:
            param = getattr(self.model, param_name)
            self.model.grads[param_name] += self.weight_decay_lambda * param
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition