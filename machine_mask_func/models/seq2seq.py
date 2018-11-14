import torch
torch.manual_seed(2)
import torch.nn as nn
import torch.nn.functional as F
import abc


class AbstractSeq2Seq:
    """
    Abstract sequence to sequence model defining common functions.
    """
    @abc.abstractmethod
    def flatten_parameters(self):
        pass

    @abc.abstractmethod
    def forward(self, *input):
        pass


class EncoderSeq2Seq(AbstractSeq2Seq, nn.Module):
    """ Sequence-to-sequence architecture only with a configurable encoder.
       Args:
           encoder (EncoderRNN): object of EncoderRNN
       Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
           - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
             each sequence is a list of token IDs. This information is forwarded to the encoder.
           - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
               in the mini-batch, it must be provided when using variable length RNN (default: `None`)
       Outputs: encoder_outputs, encoder_hidden
           - **encoder_outputs** ():
           - **encoder_hiddens** ():
       """
    def __init__(self, encoder, decode_function=F.log_softmax):
        super().__init__()
        self.encoder = encoder
        self.decode_function = decode_function
        self.functional_groups = torch.randperm(self.encoder.hidden_size).view(16,-1)
        #self.out = nn.Linear(self.encoder.rnn.hidden_size, self.encoder.output_vocab_size)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, **unused):
        encoder_outputs, encoder_hidden, encoder_masks = self.encoder(input_variable, input_lengths)
        #log_probs = self.decode_function(self.out(encoder_outputs), dim=-1)
#        other = dict()
        # other['sequence'] = log_probs.topk(1)[1] # B x S x V
        #
        # a = list()
        # b = list()
        # for si in range(other['sequence'].size(1)):
        #     a.append(other['sequence'][:, si])
        #     b.append(log_probs[:, si])
        # other['sequence'] = a
#        other['encoder_masks'] = encoder_masks
        # log_probs = b
        #result = log_probs, encoder_hidden, other

        return encoder_outputs, encoder_hidden, encoder_masks


class DecoderSeq2Seq(AbstractSeq2Seq, nn.Module):
    """ Sequence-to-sequence architecture only using a configurable decoder.
       Args:
           decoder (DecoderRNN): object of DecoderRNN
           decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
       Inputs: encoder_outputs, encoder_hidden, target_variable, teacher_forcing_ratio
           - **encoder_outputs** ():
           - **encoder_hiddens** ():
           - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
             each sequence is a list of token IDs. This information is forwarded to the decoder.
           - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
             is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
             teacher forcing would be used (default is 0)
       Outputs: decoder_outputs, decoder_hidden, ret_dict
           - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
             outputs of the decoder.
           - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
             state of the decoder.
           - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
             representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
             predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
             sequences, where each list is of attention weights }.
       """
    def __init__(self, decoder, decode_function=F.log_softmax):
        super().__init__()
        self.decoder = decoder
        self.decode_function = decode_function
        self.functional_groups = torch.randperm(self.decoder.hidden_size).view(16,-1)


    def flatten_parameters(self):
        self.decoder.rnn.flatten_parameters()

    def forward(self, encoder_hidden, encoder_outputs, target_variables=None,
                teacher_forcing_ratio=0, **kwargs):
        # Unpack target variables
        try:
            target_output = target_variables.get('decoder_output', None)
            # The attention target is preprended with an extra SOS step. We must remove this
            provided_attention = target_variables['attention_target'][:, 1:] \
                if 'attention_target' in target_variables else None
        except AttributeError:
            target_output = None
            provided_attention = None

        result = self.decoder(inputs=target_output,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              provided_attention=provided_attention)

        return result
class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super().__init__()
        self.encoder_seq2seq = EncoderSeq2Seq(encoder)
        self.decoder_seq2seq = DecoderSeq2Seq(decoder, decode_function)

    def flatten_parameters(self):
        self.encoder_seq2seq.flatten_parameters()
        self.decoder_seq2seq.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variables=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, encoder_masks  = self.encoder_seq2seq.forward(input_variable, input_lengths)
        decoder_output, decoder_hidden, ret_dict = self.decoder_seq2seq.forward(encoder_hidden, encoder_outputs, target_variables, teacher_forcing_ratio)
        ret_dict['encoder_masks'] = encoder_masks
        ret_dict['encoder_activations'] = encoder_outputs
        ret_dict['enc_fct_gr'] = self.encoder_seq2seq.functional_groups
        ret_dict['dec_fct_gr'] = self.decoder_seq2seq.functional_groups
        result = decoder_output, decoder_hidden, ret_dict
        return result