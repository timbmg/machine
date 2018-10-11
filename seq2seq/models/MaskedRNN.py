import torch
import torch.nn as nn
import torch.nn.functional as F

import math

MASK_TYPES = ['no_mask', 'feat', 'input', 'elem']
CELL_TYPES = ['srn', 'gru', 'lstm']
CONDITIONS = ['x', 'h', 'x_h']


class MaskedRNN(nn.Module):
    """
    Applies RNN to a sequence with masked parameter access.

    Args:
        input_size (int): feature size of the input (e.g. embedding size)
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int): number of recurrent layers
        batch_first (bool, optional):
        bidirectional (bool, optional):
            if True, becomes a bidirectional encoder (default False)
        dropout (float, optional):
            dropout probability for the output sequence (default: 0)
        cell_type (string, optional):
            Either 'srn' for simple recurrent NN, 'gru' for gated recurrent NN
            or 'lstm' for long short term memory NN (default 'lstm')
        mask_type (string, optional):
            Either 'no_mask' for default Linear function or 'feat' for
            feature-wise or 'input' for input-wise or 'elem' for element-wise
            masking of the input gate parameters. Else vanilla linear
            transformations are used. (default 'feat')
        mask_type_hidden (string, optional):
            Either 'no_mask' for default Linear function or 'feat' for
            feature-wise or or 'input' for input-wise 'elem' for element-wise
            masking of the hidden gate parameters. Else vanilla linear
            transformations are used. (default 'feat')
        mask_condition_input (string, optional):
            Either 'x' to condtion on input x, 'h' for hidden state or 'x_h' to
            condtion on both. Applied to input layer.
        mask_condition_hidden (string, optional):
            Either 'x' to condtion on input x, 'h' for hidden state or 'x_h' to
            condtion on both. Applied to recurrent layer.

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
        >> m_lstm = MaskedRNN(10, 5, 1)
        >> x = torch.FloatTensor(4, 8, 10)
        >> y, hx = m_lstm(x) # [4, 8, 5], [1, 4, 5]

    """

    def __init__(self, input_size, hidden_size, n_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, cell_type='lstm',
                 mask_type_input='feat', mask_type_hidden='feat',
                 mask_condition_input='x', mask_condition_hidden='h',
                 identity_connection=False):

        super(MaskedRNN, self).__init__()

        if n_layers > 1:
            raise NotImplementedError()

        if not batch_first:
            raise NotImplementedError()

        if bidirectional:
            raise NotImplementedError()

        if dropout > 0:
            raise NotImplementedError()

        if cell_type not in CELL_TYPES:
            raise ValueError("{} cell not supported.".format(cell_type))

        if mask_condition_input not in CONDITIONS:
            raise ValueError("{} condition not supported."
                             .format(mask_condition_input))
        if mask_condition_hidden not in CONDITIONS:
            raise ValueError("{} condition not supported."
                             .format(mask_condition_hidden))

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout

        self.cell = RecurrentCell(
            cell=cell_type,
            input_size=input_size,
            hidden_size=hidden_size,
            mask_type_input=mask_type_input,
            mask_type_hidden=mask_type_hidden,
            mask_condition_input=mask_condition_input,
            mask_condition_hidden=mask_condition_hidden,
            identity_connection=identity_connection,
            n_layers=n_layers,
            batch_first=self.batch_first,
            bidirectional=bidirectional)

        self.reset_parameters()

    def reset_parameters(self):
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

        output, hx, masked = self.cell(input, hx)

        # repack
        if is_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(
                output,
                lengths,
                batch_first=self.batch_first)

        return output, hx, masked


class RecurrentCell(nn.Module):
    """
    Implements different recurrent cells.

    Args:
        cell (str):
            Either 'srn' for simple recurrent NN, 'gru' for gated recurrent NN
            or 'lstm' for long short term memory NN (default 'lstm')
        input_size (int): feature size of the input (e.g. embedding size)
        hidden_size (int): the number of features in the hidden state `h`
        mask_type_input (string, optional):
            Either 'no_mask' for default Linear function or  'feat' for
            feature-wise or or 'input' for input-wise 'elem' for element
            masking of the input gate parameters. Else vanilla linear
            transformations are used. (default 'feat')
        mask_type_hidden (string, optional):
            Either 'no_mask' for default Linear function or  'feat' for
            feature-wise or or 'input' for input-wise 'elem' for element
            masking of the hidden gate parameters. Else vanilla linear
            transformations are used. (default 'feat')
        mask_condition_input (string):
            Either 'x' to condtion on input x, 'h' for hidden state or 'x_h' to
            condtion on both. Applied to input layer.
        mask_condition_hidden (string):
            Either 'x' to condtion on input x, 'h' for hidden state or 'x_h' to
            condtion on both. Applied to recurrent layer.
        identity_connection (bool):
            If true, input will be added to the output at the end of the
            computation. I.e. F(x) + x. Note, that in_features and out_features
            must be equal. Note, this will only be applied to the hidden to
            hidden linear transformation. (default: False)
        n_layers (int, optional): (default: 1)
        batch_first (bool, optional): (default: True)
        bidirectional (bool, optional):
            if True, becomes a bidirectional encoder (default: False)
    """

    def __init__(self, cell, input_size, hidden_size, mask_type_input,
                 mask_type_hidden, mask_condition_input, mask_condition_hidden,
                 identity_connection, n_layers=1, batch_first=True,
                 bidirectional=False):

        super(RecurrentCell, self).__init__()

        self.cell = cell.lower()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first

        self.mask_condition_input = mask_condition_input
        self.mask_condition_hidden = mask_condition_hidden

        mask_in_features_input, mask_in_features_hidden = 0, 0
        if 'x' in mask_condition_input:
            mask_in_features_input += input_size
        if 'h' in mask_condition_input:
            mask_in_features_input += hidden_size
        if mask_in_features_input == 0 and len(mask_condition_input) > 0:
            raise RuntimeError()

        if 'x' in mask_condition_hidden:
            mask_in_features_hidden += input_size
        if 'h' in mask_condition_hidden:
            mask_in_features_hidden += hidden_size
        if mask_in_features_hidden == 0 and len(mask_condition_hidden) > 0:
            raise RuntimeError()

        input_args = (input_size,
                      hidden_size,
                      mask_type_input,
                      mask_in_features_input)
        hidden_args = (hidden_size,
                       hidden_size,
                       mask_type_hidden,
                       mask_in_features_hidden,
                       identity_connection)

        if self.cell == 'srn':
            self.W = MaskedLinear(*input_args)
            self.U = MaskedLinear(*hidden_args)
            self.b = nn.Parameter(torch.Tensor(hidden_size))

            self.forward_step_fn = self._rnn_forward_step

        elif self.cell == 'gru':
            self.W_r = MaskedLinear(*input_args)
            self.W_z = MaskedLinear(*input_args)
            self.W_h = MaskedLinear(*input_args)

            self.U_r = MaskedLinear(*hidden_args)
            self.U_z = MaskedLinear(*hidden_args)
            self.U_h = MaskedLinear(*hidden_args)

            self.b_r = nn.Parameter(torch.Tensor(hidden_size))
            self.b_z = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h = nn.Parameter(torch.Tensor(hidden_size))

            self.forward_step_fn = self._gru_forward_step

        elif self.cell == 'lstm':
            self.W_f = MaskedLinear(*input_args)
            self.W_i = MaskedLinear(*input_args)
            self.W_o = MaskedLinear(*input_args)
            self.W_c = MaskedLinear(*input_args)

            self.U_f = MaskedLinear(*hidden_args)
            self.U_i = MaskedLinear(*hidden_args)
            self.U_o = MaskedLinear(*hidden_args)
            self.U_c = MaskedLinear(*hidden_args)

            self.b_f = nn.Parameter(torch.Tensor(hidden_size))
            self.b_i = nn.Parameter(torch.Tensor(hidden_size))
            self.b_o = nn.Parameter(torch.Tensor(hidden_size))
            self.b_c = nn.Parameter(torch.Tensor(hidden_size))

            self.forward_step_fn = self._lstm_forward_step

        else:
            raise ValueError("{} not supported.".format(self.cell))

    def forward(self, input, hx=None):

        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_size = input.size(1) if self.batch_first else input.size(0)

        if hx is None:
            hx = self._init_hidden(input, batch_size)
        else:
            if self.cell == 'lstm':
                hx = hx[0].transpose(0, 1), hx[1].transpose(0, 1)
            else:
                hx = hx.transpose(0, 1)

        output = list()
        for si in range(sequence_size):

            x = input[:, si].unsqueeze(1)

            if self.mask_condition_input == 'x':
                mask_input = x
            elif self.mask_condition_input == 'h':
                mask_input = hx[0] if self.cell == 'lstm' else hx
            elif self.mask_condition_input == 'x_h':
                mask_input = torch.cat([x, hx], dim=-1)

            if self.mask_condition_hidden == 'h':
                mask_hidden_input = hx[0] if self.cell == 'lstm' else hx
            elif self.mask_condition_hidden == 'x':
                mask_hidden_input = x
            elif self.mask_condition_hidden == 'x_h':
                mask_hidden_input = torch.cat([x, hx], dim=-1)

            hx, masks = self.forward_step_fn(x, hx, mask_input, mask_hidden_input)

            if self.cell == 'lstm':
                output.append(hx[0])
            else:
                output.append(hx)

        output = torch.cat(output, dim=1)
        if self.cell == 'lstm':
            hx = hx[0].transpose(0, 1), hx[1].transpose(0, 1)
        else:
            hx = hx.transpose(0, 1)

        return output, hx, masks

    def _init_hidden(self, input, batch_size):
        hx = input.new_zeros(
            batch_size,
            self.n_layers * 1,
            self.hidden_size,
            requires_grad=False)
        if self.cell == 'lstm':
            return (hx, hx)
        else:
            return hx

    def _rnn_forward_step(self, x, hh, mask_input, mask_hidden_input):
        hx_, mask_W = self.W(x, mask_input)
        hh_, mask_U = self.U(hh, mask_hidden_input)
        hh = F.tanh(hx_ + hh_ + self.b)
        return hh, {'mask_W': mask_W, 'mask_U': mask_U}

    def _gru_forward_step(self, x, hx, mask_input, mask_hidden_input):
        hx_r, mask_W_r = self.W_r(x, mask_input)
        hh_r, mask_U_r = self.U_r(hx, mask_hidden_input)
        r = F.sigmoid(hx_r + hh_r + self.b_r)
        hx_z, mask_W_z = self.W_z(x, mask_input)
        hh_z, mask_U_z = self.U_z(hx, mask_hidden_input)
        z = F.sigmoid( hx_z + hh_z + self.b_z)
        # NOTE: masked_hidden_input should be redefined with r*hx hidden state

        hx_h, mask_W_h = self.W_h(x, mask_input)
        hh_h, mask_U_h = self.U_h((r * hx), mask_hidden_input)
        hx = z * hx + (1-z) * F.tanh(hx_h + hh_h + self.b_h)

        return hx, {'mask_W_r': mask_W_r, 'mask_U_r': mask_U_r,
         'mask_W_z': mask_W_z, 'mask_U_z': mask_U_z, 'mask_W_h':mask_W_h, 'mask_U_h': mask_U_h}

    def _lstm_forward_step(self, x, hx, mask_input, mask_hidden_input):
        # TODO:
        h, c = hx
        f = F.sigmoid(self.W_f(x, mask_input) +
                      self.U_f(h, mask_hidden_input) +
                      self.b_f)
        i = F.sigmoid(self.W_i(x, mask_input) +
                      self.U_i(h, mask_hidden_input) +
                      self.b_i)
        o = F.sigmoid(self.W_o(x, mask_input) +
                      self.U_o(h, mask_hidden_input) +
                      self.b_o)
        c = f * c + F.tanh(self.W_c(x, mask_input) +
                           self.U_c(h, mask_hidden_input) +
                           self.b_c) * i
        h = o * c

        return (h, c)


class MaskedLinear(nn.Module):
    """Implements a linear transformation with masked parameter access.

    Args:
        in_features (int):
        out_features (int):
        wise (str):
            Either 'no_mask' for default Linear function or 'feat' for
            feature-wise or 'input' for input-wise or 'elem' for element-wise
            masking of the parameters.
        mask_in_features (int, optional): Number of input dimensions of the
            masking matrix.
        identity_connection (bool, optional):
            If true, input will be added to the output at the end of the
            computation. I.e. F(x) + x. Note, that in_features and out_features
            must be equal. (default: False)
    Inputs: input
        - **input**: torch.FloatTensor of size N*, in_features
    Outputs: output
        - **output**: torch.FloatTensor of size N*, out_features
    Examples:
        >> ml = MaskedLinear(10, 5, 'feat')
        >> x = torch.FloatTensor(4, 10)
        >> y = ml(x) # [4, 5]

    """

    def __init__(self, in_features, out_features, wise,
                 mask_in_features=None, identity_connection=False):

        super(MaskedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.wise = wise
        self.identity_connection = identity_connection
        if self.identity_connection and self.in_features != self.out_features:
            raise RuntimeError("For identity connection, in- and output " +
                               "features must be equal, but got {} and {}"
                               .format(self.in_features, self.out_features))
        if mask_in_features is None:
            self.mask_in_features = in_features
        else:
            self.mask_in_features = mask_in_features

        if wise == 'no_mask':
            self.mask_in_features = 0
            self.mask_out_features = 0
        elif wise == 'feat':
            self.mask_out_features = out_features
        elif wise == 'input':
            self.mask_out_features = in_features
        elif wise == 'elem':
            self.mask_out_features = out_features * in_features
        else:
            raise ValueError("{}-wise masking not supported. Chose from {}."
                             .format(wise, ', '.join(MASK_TYPES)))

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))

        if wise != 'no_mask':
            self.W_mask = nn.Parameter(
                torch.Tensor(self.mask_out_features, self.mask_in_features))

    def forward(self, input, mask_input=None):

        if self.wise == 'no_mask':
            output = F.linear(input, self.W)
            mask = None
        else:
            if mask_input is None:
                mask_input = input

            mask = F.sigmoid(F.linear(mask_input, self.W_mask))

            if self.wise == 'feat':
                output = mask * F.linear(input, self.W)

            elif self.wise == 'input':
                output = F.linear(mask * input, self.W)

            elif self.wise == 'elem':
                # TODO: This only works for 3D input right now
                mask = mask.view(-1, self.out_features, self.in_features)
                masked_weight = \
                    mask * self.W.unsqueeze(0).repeat(mask.size(0), 1, 1)
                output = torch.bmm(input, masked_weight.transpose(1, 2))

        if self.identity_connection:
            output += input
        return output, mask

    def __repr__(self):
        return ("MaskedLinear(in_features={}, out_features={}, wise={}, " +
                "mask_in_features={}, mask_out_features={}, identity={})")\
               .format(self.in_features,
                       self.out_features,
                       self.wise,
                       self.mask_in_features,
                       self.mask_out_features,
                       self.identity_connection)
