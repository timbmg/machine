"""
Summary
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Currently implemented as recurrent LSTM model (not seq2seq)


class Teacher(nn.Module):

    """
    Summary

    Attributes:
        embedding_dim (TYPE): Description
        gamma (TYPE): Description
        hidden (TYPE): Description
        hidden_dim (TYPE): Description
        input_embedding (TYPE): Description
        input_vocab_size (TYPE): Description
        linear_output (TYPE): Description
        lstm (TYPE): Description
        output_vocab_size (TYPE): Description
        rewards (list): Description
        saved_log_probs (list): Description
    """

    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, gamma):
        """
        Summary

        Args:
            input_vocab_size (TYPE): Description
            embedding_dim (TYPE): Description
            hidden_dim (TYPE): Description
            gamma (TYPE): Description
        """
        super(Teacher, self).__init__()

        self.gamma = gamma

        self.encoder = TeacherEncoder(
            input_vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim)

        self.decoder = TeacherDecoder(
            hidden_dim=hidden_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state, valid_action_mask, max_decoding_length):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        encoder_outputs, encoded_hidden = self.encoder(input_variable=state)
        action_probs = self.decoder(encoder_outputs=encoder_outputs, hidden=encoded_hidden, output_length=max_decoding_length, valid_action_mask=valid_action_mask)

        return action_probs

    def select_actions(self, state, input_lengths, max_decoding_length, mode):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        assert mode in ['pre_train', 'train', 'eval']

        batch_size = state.size(0)
        max_encoding_length = torch.max(input_lengths)

        # (batch_size x 1) -> (batch_size x max_encoding_length)
        input_lengths_expanded = input_lengths.unsqueeze(1).expand(-1, max_encoding_length)

        # Use arange to create list 0, 1, 2, 3, .. for each element in the batch
        encoding_steps = torch.arange(max_encoding_length).long().unsqueeze(0).expand(batch_size, -1)

        # A (batch_size x max_encoding_length) tensor that has a 1 for all valid actions and 0 for all invalid actions
        valid_action_mask = encoding_steps < input_lengths_expanded

        probabilities = self.forward(state, valid_action_mask, max_decoding_length)
        # if mode == 'train':
        #     print probabilities

        actions = []
        # TODO: Doesn't take into account mixed lengths in batch
        # Doest it matter though? Aren't extra actions/attentions just ignored?
        # decoder_sequence_length = probabilities.size(1)
        for decoder_step in range(max_decoding_length):
            probabilities_current_step = probabilities[:, decoder_step, :]

            categorical_distribution_policy = Categorical(probs=probabilities_current_step)

            import random
            sample = random.random()
            eps_threshold = 0.95

            # Pick random action
            if sample > eps_threshold and mode=='train':
                # We don't need to normalize these to probabilities, as this is already done in Categorical()
                uniform_probability_current_step = torch.autograd.Variable(valid_action_mask.float())
                categorical_distribution_uniform = Categorical(probs=uniform_probability_current_step)
                action = categorical_distribution_uniform.sample()

            # Pick policy by stochastic policy
            else:
                action = categorical_distribution_policy.sample()

            self.saved_log_probs.append(categorical_distribution_policy.log_prob(action))
             
            actions.append(action.data)

        # print actions

        return actions

    # TODO: Should we somehow make sure that this is called (at the right time)?
    # TODO: inference is not a pretty solution. For now, i don't add rewards during evaluation, which results in an error. This is a quick fix. I guess we should also provide rewards at evaluation?
    def finish_episode(self, inference_mode=False):
        """
        Summary

        Returns:
            TYPE: Description
        """
        # Calculate discounted reward of entire episode

        # We must have a reward for every action
        if inference_mode:
            del self.rewards[:]
            del self.saved_log_probs[:]
            return -1

        assert len(self.rewards) == len(self.saved_log_probs), "Number of rewards ({}) must equal number of actions ({})".format(len(self.rewards), len(self.saved_log_probs))

        R = 0
        discounted_rewards = []

        # TODO: Works, but not nice
        # Get numpy array (n_rewards x batch_size)
        import numpy
        rewards = numpy.array(self.rewards)

        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R.tolist())

        discounted_rewards = torch.Tensor(discounted_rewards)

        # TODO: This doesn't work when reward is negative, right?
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0, keepdim=True)) / \
        #     (discounted_rewards.std(dim=0, keepdim=True) + float(np.finfo(np.float32).eps))

        # (n_rewards x batch_size) -> (batch_size x n_rewards)
        discounted_rewards = discounted_rewards.transpose(0, 1)
        discounted_rewards = torch.autograd.Variable(discounted_rewards, requires_grad=False)

        # Stack list of length n_rewards with 1D Variables of length batch_size to Variable of (batch_size x n_rewards)
        saved_log_probs = torch.stack(self.saved_log_probs, dim=1)

        # Calculate policy loss
        # Multiply each reward with it's negative log-probability element-wise
        policy_loss = -saved_log_probs * discounted_rewards
        # Sum over rewards, take mean over batch
        # TODO: Should we take mean over rewards?

        policy_loss = policy_loss.sum(dim=1).mean()
        
        # policy_loss = []
        # for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
        #     policy_loss.append(-log_prob * reward)
        # policy_loss = torch.cat(policy_loss).sum()

        # Reset episode
        del self.rewards[:]
        del self.saved_log_probs[:]

        import math
        if math.isnan(policy_loss.data.numpy()[0]):
            print "NANNAN"
            exit()

        return policy_loss

class TeacherEncoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim):
        super(TeacherEncoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)


    def forward(self, input_variable):
        input_embedding = self.input_embedding(input_variable)

        # TODO: Maybe we should learn hidden0? In any case, I don't think we need this method. 
        # Pytorch initialized hidden0 to zero by default. However, is it reset after very batch?
        out, hidden = self.encoder(input_embedding)

        return out, hidden

class TeacherDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(TeacherDecoder, self).__init__()

        self.hidden_dim = hidden_dim

        # TODO: What should the input to the decoder be? Just the selected hidden state?
        # Or maybe just no input?
        # self.input_embedding = nn.Embedding(1, embedding_dim)
        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)

        self.combine_enc_seq_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    # For evaluation we need to generate an attention vector for all 50 outputs while we have only 3 inputs,
    # Thats why we use max_len. Should be fixed
    def forward(self, encoder_outputs, hidden, output_length, valid_action_mask):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        action_scores_list = []
        batch_size = encoder_outputs.size(0)

        # TODO: We use rolled out version. If we actually won't use any (informative) input to the decoder
        # we should roll it to save computation time and have cleaner code.
        for decoder_step in range(output_length):
            # TODO: What should the input to the decoder be?
            embedding = torch.autograd.Variable(torch.zeros(batch_size, 1, 1))

            out, hidden = self.decoder(embedding, hidden)

            # I copied the attention mechanism from Machine's MLP attention method
            encoder_states = encoder_outputs
            h, c = hidden
            h = h.transpose(0, 1)
            decoder_states = h
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
            mlp_output = self.combine_enc_seq_layer(mlp_input)
            mlp_output = self.activation(mlp_output)
            out = self.output_layer(mlp_output)
            action_scores = out.view(batch_size, dec_seqlen, enc_seqlen)

            action_scores_list.append(action_scores)

        # Combine the action scores for each decoder step into 1 variable
        action_scores = torch.cat(action_scores_list, dim=1)

        invalid_action_mask = valid_action_mask.ne(1).unsqueeze(1).expand(-1, output_length, -1)

        action_scores.data.masked_fill_(invalid_action_mask, -float('inf'))

        # For each decoder step, take the softmax over all actions to get probs
        # TODO: Does it make sense to use log_softmax such that we don't have to call categorical.log_prob()?
        # Would we then have to initialize Categorical with logits instead of probs?
        action_probs = F.softmax(action_scores, dim=2)

        return action_probs
