import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MaskedLinear(nn.Module):
    """Implements a linear transformation with masked parameter access.

    Args:
        in_features (int):
        out_features (int):
        wise (str): Either 'feat' for feature-wise or 'elem' for element-wise
            masking of the parameters.
    Inputs: input
        - **input**: torch.FloatTensor of size N*, in_features
    Outputs: output
        - **output**: torch.FloatTensor of size N*, out_features
    Examples:
        >> ml = MaskedLinear(10, 5, 'feat')
        >> x = torch.FloatTensor(4, 10)
        >> y = ml(x) # [4, 5]

    """

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
        self.W_mask = nn.Parameter(
            torch.Tensor(out_features*masks_per_feature, in_features))

    def forward(self, input):

        mask = F.sigmoid(F.linear(input, self.W_mask))

        if self.wise == 'feat':
            output = mask * F.linear(input, self.W)

        elif self.wise == 'elem':
            # TODO: This only works for 3D input right now
            mask = mask.view(-1, self.out_features, self.in_features)
            masked_weight = \
                mask * self.W.unsqueeze(0).repeat(mask.size(0), 1, 1)
            output = torch.bmm(input, masked_weight.transpose(1, 2))

        return output

    def __repr__(self):
        return "MaskedLinear(in_features=%i, out_features=%i, wise=%s)"\
               % (self.in_features, self.out_features, self.wise)


class MaskedLSTM(nn.Module):
    """
    Applies LSTM to a sequence with masked parameter access.

    Args:
        input_size (int): feature size of the input (e.g. embedding size)
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int): number of recurrent layers
        batch_first (bool, optional):
        bidirectional (bool, optional):
            if True, becomes a bidirectional encoder (default False)
        dropout (float, optional):
            dropout probability for the output sequence (default: 0)
        mask_input (string, optional):
            Either 'feat' for feature-wise or 'elem' for element masking of the
            input gate parameters. Else vanilla linear transformations are
            used. (default 'feat')
        mask_hidden (string, optional):
            Either 'feat' for feature-wise or 'elem'for element masking of the
            hidden gate parameters. Else vanilla linear transformations are
            used. (default 'feat')

    Inputs: input, hx
        - **input** (batch, seq_len):
            tensor containing the features of the input sequence.
        - **hx**:
            tensor containing the hidden states to initilize with. Defaults to
            zero tensor.

    Outputs: output, (h, c)
        - **output** (batch, seq_len, hidden_size):
            variable containing the output features of the input sequence
        - **hx** (num_layers * num_directions, batch, hidden_size):
            last hidden state

    Examples:
        >> m_lstm = MaskedLSTM(10, 5, 1)
        >> x = torch.FloatTensor(4, 8, 10)
        >> y, hx = m_lstm(x) # [4, 8, 5], [1, 4, 5]

    """

    def __init__(self, input_size, hidden_size, n_layers, batch_first=True,
                 bidirectional=False, dropout=0, mask_input='feat',
                 mask_hidden='feat'):

        super(MaskedLSTM, self).__init__()

        if n_layers > 1:
            raise NotImplementedError()

        if not batch_first:
            # TODO: is this ever used in machine?
            raise NotImplementedError()

        if bidirectional:
            raise NotImplementedError()

        if dropout > 0:
            raise NotImplementedError()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mask_input = mask_input
        self.mask_hidden = mask_hidden

        # settings for the linear layers
        if mask_input in ['feat', 'elem']:
            input_args = input_size, hidden_size, mask_input
            input_linear = MaskedLinear
        else:
            input_args = input_size, hidden_size
            input_linear = nn.Linear

        if mask_hidden in ['feat', 'elem']:
            hidden_args = hidden_size, hidden_size, mask_hidden
            hidden_linear = MaskedLinear
        else:
            hidden_args = hidden_size, hidden_size
            hidden_linear = nn.Linear

        self.W_f = input_linear(*input_args)
        self.W_i = input_linear(*input_args)
        self.W_o = input_linear(*input_args)
        self.W_c = input_linear(*input_args)

        self.U_f = hidden_linear(*hidden_args)
        self.U_i = hidden_linear(*hidden_args)
        self.U_o = hidden_linear(*hidden_args)
        self.U_c = hidden_linear(*hidden_args)

        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # QUESTION: this is probably not required since done in train_model.py
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for n, weight in self.named_parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):

        # deal with PackedSequence
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            _, batch_sizes = input
            input, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                input,
                batch_first=self.batch_first)

        # get sizes
        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_size = input.size(1) if self.batch_first else input.size(0)

        # initialize hidden state
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(
                batch_size,
                self.n_layers * num_directions,
                self.hidden_size,
                requires_grad=False)
            h, c = hx, hx
        else:
            h, c = hx
            h, c = h.transpose(0, 1), c.transpose(0, 1)

        # propagate input through lstm
        output = list()
        for si in range(sequence_size):

            x = input[:, si].unsqueeze(1)

            f = F.sigmoid(self.W_f(x) + self.U_f(h) + self.b_f)
            i = F.sigmoid(self.W_i(x) + self.U_i(h) + self.b_i)
            o = F.sigmoid(self.W_o(x) + self.U_o(h) + self.b_o)
            c = F.tanh(self.W_c(x) + self.U_c(h) + self.b_c) * i \
                + f * c
            h = o * c

            output.append(h)

        # collect outputs
        h, c = h.transpose(0, 1), c.transpose(0, 1)
        output = torch.cat(output, dim=1)

        # repack
        if is_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(
                output,
                lengths,
                batch_first=self.batch_first)

        return output, (h, c)
