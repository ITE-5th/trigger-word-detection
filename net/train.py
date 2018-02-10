import os
from multiprocessing.pool import Pool

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from net.network import Network

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

optimizer = Adam(net.parameters(), lr=1e-3, eps=0, weight_decay=0.01)
criterion = BCELoss()
epochs = 4000


def convert2(data: torch.FloatTensor):
    return data >= 0.5


for i in range(epochs):
    net.zero_grad()
    outputs = net(X)

    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        first = convert2(outputs.data)
        second = Y.data.byte()
        size = Y.shape[0] * Y.shape[1] * Y.shape[2]
        correct = torch.eq(first, second).sum() / size
        # correct = accuracy(Y_train, convert(outputs))
        # print("epoch %i: %f - Loss: %.4f" % (i, correct / Y_train.shape[0], loss.data[0]))
        print("Epoch[%d/%d], Loss: %.4f, Accuracy: %.4f" % (i, epochs, loss.data[0], correct / Y_train.shape[0]))
        # torch.save(net.state_dict(), os.path.join('../models/', 'net-%d.pkl' % (i)))

torch.save(net.state_dict(), os.path.join('../models/', 'net.pkl'))

# torch.save(net.state_dict(), os.path.join('./models/', 'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
