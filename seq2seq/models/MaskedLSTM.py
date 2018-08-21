import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class MaskedLinear(nn.Module):
    """
    Implements a linear transformation with masked parameter access.

    Args:
        in_features (int):
        out_features (int):
        wise (str): Either 'feat' for feature-wise or 'elem' for element-wise masking of the
            parameters.
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
        self.W_mask = nn.Parameter(torch.Tensor(out_features*masks_per_feature, in_features))

    def forward(self, input):

        mask = F.sigmoid( F.linear(input, self.W_mask) )

        if self.wise == 'feat':
            output = mask * F.linear(input, self.W)

        elif self.wise == 'elem':
            # TODO: This only works for 3D input right now
            mask = mask.squeeze(1).view(-1, self.out_features, self.in_features)
            masked_weight = mask * self.W.unsqueeze(0).repeat(mask.size(0), 1, 1)
            output = torch.bmm(input, masked_weight.transpose(1, 2))

        return output


class MaskedLSTM(nn.Module):
    """
    Applies LSTM to a sequence with masked parameter access.

    Args:
        input_size (int): feature size of the input (e.g. embedding size)
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int): number of recurrent layers
        batch_first (bool, optional):
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default False)
        dropout (float, optional): dropout probability for the output sequence (default: 0)
        mask_input (string, optional): Either 'feat' for feature-wise or 'elem' for element masking of
            the input gate parameters. Else vanilla linear transformations are used. (default 'feat')
        mask_hidden (string, optional): Either 'feat' for feature-wise or 'elem' for element masking of
            the hidden gate parameters. Else vanilla linear transformations are used. (default 'feat')

    Inputs: input, hx
        - **input** (batch, seq_len): tensor containing the features of the input sequence.
        - **hx**: tensor containing the hidden states to initilize with. Defaults to zero tensor.

    Outputs: output, (h, c)
        - **output** (batch, seq_len, hidden_size): variable containing the output features of the input sequence
        - **hx** (num_layers * num_directions, batch, hidden_size): last hidden state

    Examples:
        >> m_lstm = MaskedLSTM(10, 5, 1)
        >> x = torch.FloatTensor(4, 8, 10)
        >> y, hx = m_lstm(x) # [4, 8, 5], [1, 4, 5]

    """

    def __init__(self, input_size, hidden_size, n_layers,
        batch_first=True, bidirectional=False, dropout=0,
        mask_input='feat', mask_hidden='feat'):

        super(MaskedLSTM, self).__init__()

        if n_layers > 1:
            raise NotImplementedError()

        if batch_first == False:
            # TODO: is this ever used in machine?
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

        # initialize parameters
        gates = ['f', 'i', 'o', 'c']
        self.W, self.U, self.b = dict(), dict(), dict()
        for g in gates:
            self.W[g] = input_linear(*input_args)
            self.U[g] = hidden_linear(*hidden_args)
            self.b[g] = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):


        # deal with PackedSequence
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            _, batch_sizes = input
            input, lengths = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=self.batch_first)

        # get batch size
        batch_size = input.size(0) if self.batch_first else input.size(1)

        # initilize hidden state
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(batch_size, self.n_layers * num_directions, self.hidden_size, requires_grad=False)
            h, c = hx, hx
        else:
            h, c = hx
            h, c = h.transpose(0, 1), c.transpose(0, 1)

        # propagate input through lstm
        output = list()
        for si in range(input.size(1)):

            x = input[:, si].unsqueeze(1)

            f = F.sigmoid( self.W['f'](x) + self.U['f'](h) + self.b['f'] )
            i = F.sigmoid( self.W['i'](x) + self.U['i'](h) + self.b['i'] )
            o = F.sigmoid( self.W['o'](x) + self.U['o'](h) + self.b['o'] )
            c = f * c + F.tanh( self.W['c'](x) + self.U['c'](h) + self.b['c'] )
            h = o * c

            output.append(h)

        # collect outputs
        h, c = h.transpose(0, 1), c.transpose(0, 1)
        output = torch.cat(output, dim=1)

        # repack
        if is_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=self.batch_first)


        return output, (h, c)
