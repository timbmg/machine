import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class MaskedLinear(nn.Module):

    def __init__(self, in_features, out_features, wise):

        super(MaskedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.wise = wise

        if wise == 'feat':
            masks_per_feature = 1
        elif wise == 'elem':
            masks_per_feature = in_features
        else:
            raise ValueError()

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_mask = nn.Parameter(torch.Tensor(out_features*masks_per_feature, in_features))

    def forward(self, input):

        mask = F.sigmoid( F.linear(input, self.W_mask) )

        if self.wise == 'feat':
            out = mask * F.linear(input, self.W)

        elif self.wise == 'elem':
            mask = mask.squeeze(1).view(-1, self.out_features, self.in_features)
            masked_weight = mask * self.W.unsqueeze(0).repeat(mask.size(0), 1, 1)
            out = torch.bmm(input, masked_weight.transpose(1, 2))

        return out


class MaskedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers,
        batch_first=True, bidirectional=False, dropout=0,
        mask_input='feat', mask_hidden='feat'):

        super(MaskedLSTM, self).__init__()

        if n_layers > 1:
            raise NotImplementedError()

        if batch_first == False:
            raise NotImplementedError()

        if bidirectional == True:
            raise NotImplementedError()

        if dropout > 0:
            raise NotImplementedError()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.n_layers       = n_layers
        self.batch_first    = batch_first
        self.bidirectional  = bidirectional
        self.dropout        = dropout
        self.mask_input     = mask_input
        self.mask_hidden    = mask_hidden

        if mask_input in ['feat', 'elem']:
            input_args = input_size, hidden_size, mask_input
            linear = MaskedLinear
        else:
            input_args = input_size, hidden_size
            linear = nn.Linear

        self.W_f = linear(*input_args)
        self.W_i = linear(*input_args)
        self.W_o = linear(*input_args)
        self.W_c = linear(*input_args)

        if mask_hidden in ['feat', 'elem']:
            hidden_args = hidden_size, hidden_size, mask_input
            linear = MaskedLinear
        else:
            hidden_args = hidden_size, hidden_size
            linear = nn.Linear

        self.U_f = linear(*hidden_args)
        self.U_i = linear(*hidden_args)
        self.U_o = linear(*hidden_args)
        self.U_c = linear(*hidden_args)

        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
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

            f = F.sigmoid( self.W_f(x) + self.U_f(h) + self.b_f )
            i = F.sigmoid( self.W_i(x) + self.U_i(h) + self.b_i )
            o = F.sigmoid( self.W_o(x) + self.U_o(h) + self.b_o )
            c = f * c + F.tanh( self.W_c(x) + self.U_c(h) + self.b_c )
            h = o * c

            output.append(h)

        h, c = h.transpose(0, 1), c.transpose(0, 1)
        output = torch.cat(output, dim=1)

        if is_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=self.batch_first)

        return output, (h, c)
