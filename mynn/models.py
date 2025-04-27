from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A simple CNN model with one convolutional layer followed by two linear layers.
    """
    def __init__(self):
        super().__init__()

        # Defining the layers:
        self.conv1 = conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # Convolutional layer
        self.relu1 = ReLU()  # Activation after convolutional layer
        self.fc1 = Linear(in_dim=4 * 28 * 28, out_dim=600)  # First linear layer
        self.relu2 = ReLU()  # Activation after the first linear layer
        self.fc2 = Linear(in_dim=600, out_dim=10)  # Output to 10 classes (for classification)

        self.layers = [self.conv1, self.relu1, self.fc1, self.relu2, self.fc2]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = self.conv1(X)  # First convolutional layer
        out = self.relu1(out)  # Activation after convolutional layer
        out = out.reshape(out.shape[0], -1)  # Flatten the output from convolutional layer for the first linear layer
        out = self.fc1(out)  # First linear layer
        out = self.relu2(out)  # Activation after the first linear layer
        out = self.fc2(out)  # Second linear layer (output layer)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def save_model(self, path):
        params = [
            {'W': self.conv1.W, 'b': self.conv1.b},
            {'W': self.fc1.W, 'b': self.fc1.b},
            {'W': self.fc2.W, 'b': self.fc2.b},
        ]
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self.conv1.W = params[0]['W']
        self.conv1.b = params[0]['b']
        self.fc1.W = params[1]['W']
        self.fc1.b = params[1]['b']
        self.fc2.W = params[2]['W']
        self.fc2.b = params[2]['b']
