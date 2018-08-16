import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class GatedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, batch_first=True, bidirectional=False, dropout=0):

        super().__init__()

        if n_layers > 1:
            raise NotImplementedError()

        if batch_first == False:
            raise NotImplementedError()

        if bidirectional == True:
            raise NotImplementedError()

        if dropout > 0:
            raise NotImplementedError()

        if wise not in ['feature', 'element']:
            raise ValueError()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.batch_first    = batch_first
        self.bidirectional  = bidirectional
        self.dropout        = dropout
        self.wise           = wise

        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c = nn.Linear(input_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


    def forward(self, input, hx=None):

        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            _, batch_sizes = input
            input, lengths = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=self.batch_first)

        batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(batch_size, self.n_layers * num_directions, self.hidden_size, requires_grad=False)
            h, c = hx, hx
        else:
            h, c = hx
            h, c = h.transpose(0, 1), c.transpose(0, 1)

        output = list()
        for si in range(input.size(1)):

            x = input[:, si].unsqueeze(1)

            f = F.sigmoid(self.W_f(x) + self.U_f(h) + self.b_f)
            i = F.sigmoid(self.W_i(x) + self.U_i(h) + self.b_i)
            o = F.sigmoid(self.W_o(x) + self.U_o(h) + self.b_o)
            c = f * c + F.tanh(self.W_c(x) + self.U_c(h) + self.b_c)
            h = o * c

            output.append(h)

        h, c = h.transpose(0, 1), c.transpose(0, 1)
        output = torch.cat(output, dim=1)

        if is_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=self.batch_first)

        return output, (h, c)
