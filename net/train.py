import os

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.trigger_dataset import TriggerDataset
from net.network import Network


def save_checkpoint(state, epoch):
    torch.save(state, "../models/checkpoint-{}.pth.tar".format(epoch))


net = Network()

lr = 1e-4
eps = 1e-7
weight_decay = 0.01

params = []
for key, value in dict(net.named_parameters()).items():
    if value.requires_grad:
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay, 'eps': eps}]

# , weight_decay=0.01
# optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, eps=1e-7, weight_decay=weight_decay)
optimizer = Adam(params)
criterion = BCELoss(size_average=False)

pretrained = True

start = 0
epochs = 300

if pretrained:
    start = 601
    epochs = start + 200
    state = net.load(start)
    optimizer.load_state_dict(state['optimizer'])

net = torch.nn.DataParallel(net).cuda()

partitions = ['../dataset/partitions/partition-{}.pkl'.format(i) for i in range(4)]

batch_size = 24
dataset = TriggerDataset(partitions)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
print("Dataset Size: {}".format(len(dataset)))

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
        #
        # for param in net.parameters():
        #     print(param.grad.data.sum())
        #
        # # start debugger
        # import pdb
        #
        # pdb.set_trace()

        optimizer.step()

        if epoch % 5 == 0:
            first = outputs.data >= 0.5
            second = target.data.byte()
            size = target.shape[1] * target.shape[2]
            correct = torch.eq(first, second).sum() / size
            total_correct += correct

    if epoch % 5 == 0:
        print("Epoch[%d/%d], Loss: %.4f, Accuracy: %.4f" % (epoch, epochs, total_loss, total_correct / len(dataset)))

    if epoch >= 100 and epoch % 50 == 0:
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
