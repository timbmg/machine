import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output
        method(str): The method to compute the alignment, mlp or dot

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
        method (torch.nn.Module): layer that implements the method of computing the attention vector

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim, method):
        super(Attention, self).__init__()
        self.mask = None
        self.method = self.get_method(method, dim)

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, decoder_states, encoder_states, step):

        batch_size = decoder_states.size(0)
        decoder_states_size = decoder_states.size(2)
        input_size = encoder_states.size(1)

        # compute mask
        mask = encoder_states.eq(0.)[:,:,:1].transpose(1,2).data

        # compute attention vals
        attn = self.method(decoder_states, encoder_states, step)
        attn_before = attn.data.clone()

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        # apply local mask
        attn.data.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_states)

        return context, attn

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'mlp':
            method = MLP(dim)
        elif method == 'concat':
            method = Concat(dim)
        elif method == 'dot':
            method = Dot()
        elif method == 'hard':
            method = HardCoded()
        else:
            return ValueError("Unknown attention method")

        return method


class Concat(nn.Module):
    """
    Implements the computation of attention by applying an
    MLP to the concatenation of the decoder and encoder
    hidden states.
    """
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.mlp = nn.Linear(dim*2, 1)

    def forward(self, decoder_states, encoder_states, step):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _                = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, decoder_states, encoder_states, step):
        attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
        return attn


class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(dim*2, dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim, 1)

    def forward(self, decoder_states, encoder_states, step):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _                = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and reshape to get in correct form
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.activation(mlp_output)
        out = self.out(mlp_output)
        attn = out.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class HardCoded(nn.Module):

    """
    Hardcoded attention guidance module for the lookup table task (diagonal attentive guidance)
    """

    def forward(self, decoder_states, encoder_states, step):
        """
        Forward pass method. Computes the hard-coded, non-differentiable attentive guidance for
        the lookup tables task. Or any other task that requires a diagonal attentive guidance pattern.
        
        Args:
            decoder_states (torch.autograd.Variable): Variable containing one or multiple decoder hidden states (batch_size X dec_seqlen X hidden_size)
            encoder_states (torch.autograd.Variable): Variable containing all encoder hidden states (batch_size X enc_seqlen X hidden_size)
            step (int): Current step of the decoder. Set to -1 if the decoder is not unrolled.
        
        Returns:
            attn: The attention distribution over the encoder states for each of the provided decoder states (batch_size X dec_seqlen X enc_seqlen)
        """
        
        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _                = decoder_states.size()

        # Add hard-coded attention vector for all decoder steps
        if step == -1:
            # Initialize all-zero attention vectors
            attn = torch.zeros(batch_size, dec_seqlen, enc_seqlen)

            # In inference mode, we have more decoder steps than encoder steps. These extra steps
            # are ignored when calculating losses and metrics. For the first 'enc_seqlen' decoder steps
            # we generate the diagonal attentive guidance. For all possible steps after that, we just attend to the first encoder state.
            indices = torch.cat((
                torch.arange(enc_seqlen).view(1, enc_seqlen, 1),
                torch.zeros(1, dec_seqlen - enc_seqlen, 1)), dim=1)
            indices = indices.expand(batch_size, dec_seqlen, 1).long()

            # Fill the attention guidance with 1's
            attn = attn.scatter_(dim=2, index=indices, value=1)

        # Add hard-coded attention vector for only one single unrolled decoder step
        else:
            # Step should only be passed if we are truly unrolling the decoder and processing it
            # one step at a time.
            assert dec_seqlen == 1, "Decoder must be unrolled if 'step' is passed"

            # Initialize all-zero attention vectors
            attn = torch.zeros(batch_size, 1, enc_seqlen)

            # In inference mode, we will have to calculate attention vectors for decoding steps
            # longer than the input length. This will not be included in the calculation of metric and losses
            # We just return all-zero attention vectors
            if step < enc_seqlen:
                # Fill the attention vectors with a 1 at the specified indices/step
                indices = step * torch.ones(batch_size, 1, 1).long()
                attn = attn.scatter_(dim=2, index=indices, value=1)

        # Convert into non-grad Variable
        attn = torch.autograd.Variable(attn, requires_grad=False)

        return attn