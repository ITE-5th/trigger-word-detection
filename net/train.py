import os

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from net.network import Network


def o(x):
    return 1 if x > 0.5 else 0


def convert(x: torch.autograd.Variable):
    a = x.cpu().data.numpy()
    k = np.vectorize(o)
    return k(a)


def accuracy(x, y):
    c = 0
    for sample1, sample2 in zip(x, y):
        if np.array_equal(sample1, sample2):
            c = c + 1
    return c


X_train = np.load("../dataset/X.npy").astype(np.float32)
Y_train = np.load("../dataset/Y.npy").astype(np.float32)
# X = np.load("../datasets/X_dev.npy").astype(np.float32)
# Y = np.load("../datasets/Y_dev.npy").astype(np.float32)

X_train = np.swapaxes(X_train, 1, 2)
X = torch.autograd.Variable(torch.from_numpy(X_train)).cuda()
Y = torch.autograd.Variable(torch.from_numpy(Y_train)).cuda()

# X_dev = np.load("./dataset/X_dev.npy").astype(np.float32)
# Y_dev = np.load("./dataset/Y_dev.npy").astype(np.float32)

net = Network().cuda()

optimizer = Adam(net.parameters(), lr=0.005, weight_decay=0.01)
criterion = BCELoss().cuda()

for i in range(10001):
    net.zero_grad()

    outputs = net(X)

    loss = criterion(outputs, Y)
    loss.backward()

    optimizer.step()

    if i % 1000 == 0:
        correct = accuracy(Y_train, convert(outputs))
        print("epoch %i: %f" % (i, correct / Y_train.shape[0]))
        torch.save(net.state_dict(), os.path.join('../models/', 'net-%d.pkl' % (i)))

torch.save(net.state_dict(), os.path.join('../models/', 'net.pkl'))

# torch.save(net.state_dict(), os.path.join('./models/', 'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
