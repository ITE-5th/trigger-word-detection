import os
import time

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.trigger_dataset import TriggerDataset
from metrics.binary_accuracy import BinaryAccuracy
from metrics.composer import MetricsComposer
from metrics.f1_score import F1Score
from metrics.precision_recall import PrecisionRecall
from net.network import Network


def save_checkpoint(state, epoch):
    torch.save(state, "../models/checkpoint-{}.pth.tar".format(epoch))


lr, eps, weight_decay = 1e-4, 1e-7, 0.01
precision_threshold, recall_threshold = 70, 70

net = Network()
params = []
for key, value in dict(net.named_parameters()).items():
    if value.requires_grad:
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay, 'eps': eps}]

# optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, eps=1e-7, weight_decay=weight_decay)
optimizer = Adam(params)
criterion = BCELoss(size_average=False)


accuracy_metric = BinaryAccuracy()
precision_recall_metric = PrecisionRecall()
f1_metric = F1Score()
composer = MetricsComposer([accuracy_metric, precision_recall_metric, f1_metric])

start = 0
epochs = 200
pretrained = False
if pretrained:
    start = 357
    epochs = start + 200
    state = net.load(start)
    optimizer.load_state_dict(state['optimizer'])

net = torch.nn.DataParallel(net).cuda()

batch_size = 24
# dataset = TriggerDataset("../dataset/partitions/partition-0.pkl")
dataset = TriggerDataset("../dataset/training_set.pkl")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
print("Dataset Size: {}".format(len(dataset)))

batch_loss, total_loss = 0, 0
batches = len(dataset) / batch_size

print("Begin Training")
for epoch in range(start, epochs):
    since = time.time()
    composer.reset()

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

        # if epoch % 5 == 0:
        composer(outputs, target)

        if precision_recall_metric > (precision_threshold, recall_threshold):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, epoch)

    time_elapsed = time.time() - since
    time_str = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print("Epoch[%d/%d], Time: %s, Loss: %.4f, %s, %s, %s" % (
        epoch, epochs, time_str, total_loss, accuracy_metric, precision_recall_metric, f1_metric))

    if epoch % 50 == 0:
        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, epoch)

save_checkpoint({
    'epoch': epoch,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict()
}, epoch)