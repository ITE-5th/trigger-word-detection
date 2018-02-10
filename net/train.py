import os

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset import TriggerDataset
from net.network import Network


def save_checkpoint(state, epoch):
    torch.save(state, "../models/checkpoint-{}.pth.tar".format(epoch))


# X_train = np.load("../dataset/X.npy").astype(np.float32)
# Y_train = np.load("../dataset/Y.npy").astype(np.float32)
# # X = np.load("../datasets/X_dev.npy").astype(np.float32)
# # Y = np.load("../datasets/Y_dev.npy").astype(np.float32)
#
# X_train = np.swapaxes(X_train, 1, 2)
# X = torch.autograd.Variable(torch.from_numpy(X_train)).cuda()
# Y = torch.autograd.Variable(torch.from_numpy(Y_train)).cuda()
#
# # X_dev = np.load("./dataset/X_dev.npy").astype(np.float32)
# # Y_dev = np.load("./dataset/Y_dev.npy").astype(np.float32)

net = Network()
net = torch.nn.DataParallel(net).cuda()

pretrained = False

start = 0
epochs = 300
if pretrained:
    start = 190
    net.load_state_dict(torch.load(os.path.join('../models/net-' + str(start) + '.pkl')))
    epochs = 200 + start

optimizer = Adam(net.parameters(), lr=1e-4, eps=0, weight_decay=0.01)
criterion = BCELoss()

batch_size = 24
dataset = TriggerDataset('../dataset/dataset.pkl')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

batch_loss, total_loss = 0, 0
batches = len(dataset) / batch_size

print("Begin Training")
for epoch in range(start, epochs):

    total_loss, total_correct = 0, 0
    for batch, samples in enumerate(loader):
        inputs = torch.autograd.Variable(samples[0].float().cuda())
        target = torch.autograd.Variable(samples[1].float().cuda()).permute(0, 2, 1)

        net.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, target)
        total_loss += loss.data[0]

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            first = outputs.data >= 0.5
            second = target.data.byte()
            size = target.shape[1] * target.shape[2]
            correct = torch.eq(first, second).sum() / size
            total_correct += correct

    if epoch % 5 == 0:
        print("Epoch[%d/%d], Loss: %.4f, Accuracy: %.4f" % (epoch, epochs, total_loss, total_correct / len(dataset)))

    if epoch > 100 and epoch % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, epoch + 1)

save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict()
}, epoch + 1)

# torch.save(net.state_dict(), os.path.join('./models/', 'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
