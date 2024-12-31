# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x*y

    def backward(self, dout):
        return self.y * dout,self.x * dout

class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout):
        return dout, dout

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x * (x > 0)

    def backward(self, dout):
        return dout * (self.x > 0)

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out) * self.out

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def update(self, rate):
        self.W -= rate * self.dW
        self.b -= rate * self.db

class Softmax:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout):
        return dout, dout

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None    # softmax的输出
        self.t = None    # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dout * dx

class Network3Layer:
    def __init__(self, input_size, hidden_size1, output_size):
        self.params = {}
        self.params['W1'] = np.sqrt(2 / input_size) * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.random.randn(hidden_size1)
        self.params['W2'] = np.sqrt(2 / hidden_size1) * np.random.randn(hidden_size1, output_size)
        self.params['b2'] = np.random.randn(output_size)
        # self.params['W3'] = np.sqrt(2 / hidden_size2) * np.random.randn(hidden_size2, output_size)
        # self.params['b3'] = np.random.randn(output_size)
        # self.Layer = OrderedDict()
        self.Layer = {}
        self.Layer['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # self.Layer['Affine1'] = Affine(np.zeros([input_size, hidden_size1]), np.zeros(hidden_size1))
        self.Layer['ReLU1'] = ReLU()
        self.Layer['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # self.Layer['Affine2'] = Affine(np.zeros([hidden_size1, hidden_size2]), np.zeros(hidden_size2))
        # self.Layer['ReLU2'] = ReLU()
        # self.Layer['Affine3'] = Affine(self.params['W3'], self.params['b3'] )
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for i in self.Layer.values():
            x = i.forward(x)

        # return self.lastLayer.forward(x)
        return x

    def gradient(self, x, t):
        # forword
        for i in self.Layer.values():
            x = i.forward(x)
        self.lastLayer.forward(x, t)
        # backward
        grad = self.lastLayer.backward(1)
        LayerList = list(self.Layer.values())
        LayerList.reverse()
        for l in LayerList:
            grad = l.backward(grad)

        grads = {}
        grads['W1'] = self.Layer['Affine1'].dW
        grads['b1'] = self.Layer['Affine1'].db
        grads['W2'] = self.Layer['Affine2'].dW
        grads['b2'] = self.Layer['Affine2'].db
        # grads['W3'] = self.Layer['Affine3'].dW
        # grads['b3'] = self.Layer['Affine3'].db
        # print(grads['b1'][0])
        return grads
        # for i in self.Layer.values():
        #     if isinstance(i, Affine):
        #         i.update(learing_rate)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def loss(self, y, t):
        d = 1e-7
        # y = softmax(y)
        # if y.ndim == 1:
        #     t = t.reshape(1, t.size)
        #     y = y.reshape(1, y.size)
        return self.lastLayer.forward(y,t)

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.001, momentum=0.8):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in params.keys():
                self.h[key] = np.zeros_like(params[key])
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

def cross_entropy_error(y, t):
    # y:概率
    d = 1e-7
    # if y.ndim == 1:
    #     t = t.reshape(1, t.size)
    #     y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y + d) * t) / batch_size

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # print(t_train.shape, t_test.shape)
    return x_train, t_train, x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

x, t_, x_test, t_test = get_data()
t = np.zeros([t_.shape[0], 10])
for i in range(t_.shape[0]):
    t[i][t_[i]] = 1
test = np.zeros([t_test.shape[0], 10])
for i in range(test.shape[0]):
    test[i][t_[i]] = 1
# network = Network3(init_network())
network = Network3Layer(784, 100, 10)
batch_size = 100 # 批数量
# accuracy_cnt = 0
loss_ave = 0
# optimizer = Momentum(0.0001, 0.8)
# optimizer = AdaGrad(0.01)
optimizer = Momentum(0.001, 0.8)
for j in range(100):
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        t_batch = t[i:i + batch_size]
        y_batch = network.predict(x_batch)  # score
        loss_ave = network.loss(y_batch, t_batch)
        grads = network.gradient(x_batch, t_batch)
        # for i in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        #     network.params[i] -= 0.001 * grads[i]
        optimizer.update(network.params, grads)
    print(j, loss_ave)
    # print('accuracy: {}'.format(network.accuracy(x_test, test)))
    print('accuracy: {}'.format(network.accuracy(x, t)))

with open("2layerNet.pkl", "wb") as f:
    pickle.dump(network, f)