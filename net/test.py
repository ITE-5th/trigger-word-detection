import os

import torch
from torch.utils.data import DataLoader

from dataset.trigger_dataset import TriggerDataset
from net.network import Network


net = Network()
state = net.load(601)
net = torch.nn.DataParallel(net).cuda()

partition = 4

batch_size = 24
dataset = TriggerDataset('../dataset/partitions/partition-{}.pkl'.format(partition))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

batches = len(dataset) / batch_size

print("Begin Testing")
total_correct = 0
for batch, samples in enumerate(loader):
    inputs = torch.autograd.Variable(samples[0].float().cuda())
    target = torch.autograd.Variable(samples[1].float().cuda()).permute(0, 2, 1)

    net.zero_grad()
    outputs = net(inputs)

    first = outputs.data >= 0.5
    second = target.data.byte()
    size = target.shape[1] * target.shape[2]
    correct = torch.eq(first, second).sum() / size
    total_correct += correct

print("Test Accuracy: %.4f" % (total_correct / len(dataset)))

# torch.save(net.state_dict(), os.path.join('./models/', 'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
