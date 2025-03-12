class Dense:
    def __init__(self, input_dim, output_dim, activation_function):
        self.synapse = 2 * np.random.random((input_dim, output_dim)) - 1
        self.select_activation_function(activation_function)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, y):
        return 1 - y ** 2

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dsigmoid(self, y):
        return y*(1-y)

    def relu(self, x):
        return x * (x > 0)

    def drelu(self, x):
        return 1. * (x > 0)
    
    def select_activation_function(self, activation_function):
        if activation_function == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.dtanh
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.dsigmoid
        if activation_function == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.drelu
        if activation_function == "linear":
            self.activation_function = lambda x: x
            self.activation_derivative = lambda x: x

    def forward(self, previous_layer):
        self.output = self.activation_function(
            previous_layer.dot(self.synapse)
        )
        return self.output
                           
    def compute_gradient(self, layer, error):
        self.delta = error * self.activation_derivative(layer)
        return self.delta.dot(self.synapse.T)

    def prepare_for_multiplication(self, vector):
        num_cols = len(vector)
        num_rows = 1
        return vector.reshape(num_rows, num_cols)

    def update_weights(self, layer, learning_rate):
        layer = self.prepare_for_multiplication(layer)
        delta = self.prepare_for_multiplication(self.delta)
        self.synapse += layer.T.dot(delta) * learning_rate
