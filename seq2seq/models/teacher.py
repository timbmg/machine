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

    def forward(self, state, max_len):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        encoder_outputs, encoded_hidden = self.encoder(input_variable=state)
        action_probs = self.decoder(encoder_outputs=encoder_outputs, hidden=encoded_hidden, output_length=max_len)

        return action_probs

    def select_actions(self, state, max_len):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        enc_len = state.size(1)
        probabilities = self.forward(state, max_len)

        # print probabilities

        import random
        sample = random.random()
        eps_threshold = 0.95

        actions = []
        # TODO: Doesn't take into account mixed lengths in batch
        # Doest it matter though? Aren't extra actions/attentions just ignored?
        # decoder_sequence_length = probabilities.size(1)
        for decoder_step in range(max_len):
            probabilities_current_step = probabilities[:, decoder_step, :]

            categorical_distribution = Categorical(probs=probabilities_current_step)

            if sample > eps_threshold:
                action = torch.autograd.Variable(torch.LongTensor([random.randrange(enc_len)]))
            else:
                action = categorical_distribution.sample()

            self.saved_log_probs.append(categorical_distribution.log_prob(action))
            actions.append(action.data[0])

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
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.Tensor(discounted_rewards)
        # TODO: This doesn't work when reward is negative, right?
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
            (discounted_rewards.std() + np.finfo(np.float32).eps)

        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Reset episode
        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss

class TeacherEncoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim):
        super(TeacherEncoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)


    # TODO: Maybe we should learn hidden0? In any case, I don't think we need this method. 
    # Pytorch initialized hidden0 to zero by default. However, is it reset after very batch?
    def init_hidden(self):
        """
        Summary

        Returns:
            TYPE: Description
        """
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        hidden0 = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                   torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

        return hidden0

    def forward(self, input_variable):
        input_embedding = self.input_embedding(input_variable)

        hidden0 = self.init_hidden()
        out, hidden = self.encoder(input_embedding, hidden0)

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
    def forward(self, encoder_outputs, hidden, output_length):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        action_scores_list = []

        # TODO: We use rolled out version. If we actually won't use any (informative) input to the decoder
        # we should roll it to save computation time and have cleaner code.
        for decoder_step in range(output_length):
            # TODO: What should the input to the decoder be?
            embedding = torch.autograd.Variable(torch.zeros(1,1,1))

            out, hidden = self.decoder(embedding, hidden)

            # I copied the attention mechanism from Machine's MLP attention method
            encoder_states = encoder_outputs
            h, c = hidden
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

        # For each decoder step, take the softmax over all actions to get probs
        # TODO: Does it make sense to use log_softmax such that we don't have to call categorical.log_prob()?
        # Would we then have to initialize Categorical with logits instead of probs?
        action_probs = F.softmax(action_scores, dim=2)

        return action_probs
