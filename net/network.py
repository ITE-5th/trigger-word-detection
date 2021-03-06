import torch
from torch.autograd import Variable
from torch.nn import *

from wrappers.time_distributed import TimeDistributed


class Network(Module):

    def __init__(self):
        super().__init__()

        self.conv = Conv1d(101, 196, 15, stride=4)
        self.batch_norm_0 = BatchNorm1d(196, momentum=0.99, eps=0.001)
        self.relu = ReLU()
        self.dropout_0 = Dropout(0.5)

        self.gru_1 = GRU(input_size=196, hidden_size=128, batch_first=True)
        self.dropout_1 = Dropout(0.5)
        self.batch_norm_1 = BatchNorm1d(128, momentum=0.99, eps=0.001)

        self.gru_2 = GRU(input_size=128, hidden_size=128, batch_first=True)
        self.dropout_21 = Dropout(0.5)
        self.batch_norm_2 = BatchNorm1d(128, momentum=0.99, eps=0.001)
        self.dropout_22 = Dropout(0.5)
        self.time_distributed = TimeDistributed(torch.nn.Sequential(Linear(128, 1), Sigmoid()), batch_first=True)
        # self.time_distributed = TimeDistributed(Linear(128, 1), batch_first=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.batch_norm_0(x)
        x = self.relu(x)
        x = self.dropout_0(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru_1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.dropout_1(x)
        x = self.batch_norm_1(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru_2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.dropout_21(x)
        x = self.batch_norm_2(x)
        x = self.dropout_22(x)

        x = x.permute(0, 2, 1)
        x = self.time_distributed(x)

        return x

    def predict(self, inputs):
        inputs = Variable(torch.from_numpy(inputs))
        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()

        x = self.conv(inputs)
        x = self.batch_norm_0(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru_1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.batch_norm_1(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru_2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.batch_norm_2(x)

        x = x.permute(0, 2, 1)
        x = self.time_distributed(x)

        return x

    def load(self, epoch):
        state = torch.load("../models/checkpoint-{}.pth.tar".format(epoch))
        state_dict = state['state_dict']

        new_state_dict = dict()
        for key in state_dict.keys():
            new_name = key[key.index(".") + 1:]
            new_state_dict[new_name] = state_dict[key]

        self.load_state_dict(new_state_dict)

        return state
