# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from common.util import im2col, col2im

def cross_entropy_error(y, t):
    # y:概率
    d = 1e-7
    # if y.ndim == 1:
    #     t = t.reshape(1, t.size)
    #     y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y + d) * t) / batch_size

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
    # print(t_train.shape, t_test.shape)
    return x_train, t_train, x_test, t_test

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x * (x > 0)

    def backward(self, dout):
        return dout * (self.x > 0)

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(self.original_x_shape)
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

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        OW = int((W + 2*self.pad - FW) / self.stride + 1)
        OH = int((H + 2*self.pad - FH) / self.stride + 1)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 滤波器的展开
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, h, w, stride=1, pad=0):
        self.h = h
        self.w = w
        self.stride = stride
        self.pad = pad
        self.arg_max = None
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        self.shape = x.shape
        OW = int((W + 2 * self.pad - self.w) / self.stride + 1)
        OH = int((H + 2 * self.pad - self.h) / self.stride + 1)
        col = im2col(x, self.h, self.w, self.stride, self.pad).reshape(-1, self.h*self.w)
        out = np.max(col, axis=1)
        self.arg_max = np.argmax(col, axis=1)
        self.x = x
        # print(self.maxIndex)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.h * self.w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.h, self.w, self.stride, self.pad)

        return dx

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=1):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size + filter_pad*2 - filter_size)/filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) *
                               (conv_output_size / 2))
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0],
                                            filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size,
                                            hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = {}
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = ReLU()
        self.layers['Pool1'] = Pooling(h=2, w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                      self.params['b2'])
        self.layers['Relu2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                      self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        out = x
        for layer in self.layers.values():
            out = layer.forward(out)

        return out

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backword
        layers = list(self.layers.values())
        layers.reverse()
        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

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

x, t_, x_test, t_test = get_data()
x = x[0:500]
t = np.zeros([t_.shape[0], 10])
for i in range(t_.shape[0]):
    t[i][t_[i]] = 1
t = t[0:500]
test = np.zeros([t_test.shape[0], 10])
for i in range(test.shape[0]):
    test[i][t_[i]] = 1
# network = Network3(init_network())
network = SimpleConvNet()
batch_size = 100 # 批数量
# accuracy_cnt = 0
loss_ave = 0
# optimizer = Momentum(0.0001, 0.8)
# optimizer = AdaGrad(0.01)
optimizer = Momentum(0.002, 0.8)
for j in range(100):
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        t_batch = t[i:i + batch_size]
        # y_batch = network.predict(x_batch)  # score
        loss_ave = network.loss(x_batch, t_batch)
        grads = network.gradient(x_batch, t_batch)
        # for i in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        #     network.params[i] -= 0.001 * grads[i]
        optimizer.update(network.params, grads)
       # print(1)
    print(j, loss_ave)
    # print('accuracy: {}'.format(network.accuracy(x_test, test)))
    print('accuracy: {}'.format(network.accuracy(x, t)))

