from __future__ import print_function, division

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Teacher(nn.Module):

    """
    Seq2seq understander model with attention. Trained using reinforcement learning.
    First, pass the input sequence to `select_actions()` to perform forward pass and retrieve the actions
    Next, calculate and pass the rewards for the selected actions.
    Finally, call `finish_episod()` to calculate the discounted rewards and policy loss.
    """

    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, gamma):
        """
        Args:
            input_vocab_size (int): Total size of the input vocabulary
            embedding_dim (int): Number of units to use for the input symbol embeddings
            hidden_dim (int): Size of the RNN cells in both encoder and decoder
            gamma (float): Gamma value to use for the discounted rewards
        """
        super(Teacher, self).__init__()

        self.encoder = TeacherEncoder(
            input_vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim)

        self.decoder = TeacherDecoder(
            hidden_dim=hidden_dim)

        self.gamma = gamma

        self._saved_log_probs = []
        self._rewards = []

    def forward(self, state, valid_action_mask, max_decoding_length):
        """
        Perform a forward pass through the seq2seq model

        Args:
            state (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence
            valid_action_mask (torch.tensor): [batch_size x max_input_length] ByteTensor containing a 1 for all non-pad inputs
            max_decoding_length (int): Maximum length till which the decoder should run

        Returns:
            torch.tensor: [batch_size x max_output_length x max_input_length] tensor containing the probabilities for each decoder step to attend to each encoder step
        """
        encoder_outputs, encoded_hidden = self.encoder(input_variable=state)
        action_probs = self.decoder(encoder_outputs=encoder_outputs, hidden=encoded_hidden,
                                    output_length=max_decoding_length, valid_action_mask=valid_action_mask)

        return action_probs

    def select_actions(self, state, input_lengths, max_decoding_length, epsilon):
        """
        Perform forward pass and stochastically select actions using epsilon-greedy RL

        Args:
            state (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence
            input_lengths (list): List containing the input length for each element in the batch
            max_decoding_length (int): Maximum length till which the decoder should run
            epsilon (float): epsilon for epsilon-greedy RL. Set to 1 in inference mode

        Returns:
            list(torch.tensor): List of length max_output_length containing the selected actions
        """
        if self._rewards:
            raise Exception("Did you forget to finish the episode?")

        batch_size = state.size(0)

        # First, we establish which encoder states are valid to attend to. For
        # this we use the input_lengths
        max_encoding_length = torch.max(input_lengths)

        # (batch_size) -> (batch_size x max_encoding_length)
        input_lengths_expanded = input_lengths.unsqueeze(1).expand(-1, max_encoding_length)

        # Use arange to create list 0, 1, 2, 3, .. for each element in the batch
        # (batch_size x max_encoding_length)
        encoding_steps_indices = torch.arange(max_encoding_length, dtype=torch.long, device=device)
        encoding_steps_indices = encoding_steps_indices.unsqueeze(0).expand(batch_size, -1)

        # A (batch_size x max_encoding_length) tensor that has a 1 for all valid
        # actions and 0 for all invalid actions
        valid_action_mask = encoding_steps_indices < input_lengths_expanded

        # We perform a forward pass to get the probability of attending to each
        # encoder for each decoder
        probabilities = self.forward(state, valid_action_mask, max_decoding_length)

        actions = []
        for decoder_step in range(max_decoding_length):
            # Get the probabilities for a single decoder time step
            # (batch_size x max_encoder_states)
            probabilities_current_step = probabilities[:, decoder_step, :]

            categorical_distribution_policy = Categorical(probs=probabilities_current_step)

            # Perform epsilon-greed action sampling
            sample = random.random()
            # If we don't meet the epsilon threshold, we stochastically sample from the policy
            if sample <= epsilon:
                action = categorical_distribution_policy.sample()
            # Else we sample the actions from a uniform distribution (over the valid actions)
            else:
                # We don't need to normalize these to probabilities, as this is already
                # done in Categorical()
                uniform_probability_current_step = valid_action_mask.float()
                categorical_distribution_uniform = Categorical(
                    probs=uniform_probability_current_step)
                action = categorical_distribution_uniform.sample()

            # Store the log-probabilities of the chosen actions
            self._saved_log_probs.append(categorical_distribution_policy.log_prob(action))

            actions.append(action)

        # Convert list into tensor and make it batch-first
        actions = torch.stack(actions).transpose(0, 1)

        return actions

    def set_rewards(self, rewards):
        self._rewards = rewards

    def finish_episode(self):
        """
        Calculate discounted reward of entire episode and return policy loss


        Returns:
            torch.tensor: Single float representing the policy loss
        """

        # In inference mode, no rewards are added and we don't have to do anything besides reset.
        if not self._rewards:
            del self._rewards[:]
            del self._saved_log_probs[:]
            return None

        assert len(self._rewards) == len(self._saved_log_probs), "Number of rewards ({}) must equal number of actions ({})".format(len(self._rewards), len(self._saved_log_probs))

        # Calculate discounted rewards
        R = 0
        discounted_rewards = []

        # Get numpy array (n_rewards x batch_size)
        rewards = np.array(self._rewards)

        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R.tolist())

        discounted_rewards = torch.tensor(discounted_rewards, requires_grad=False, device=device)

        # TODO: This doesn't work when reward is negative
        # Normalize rewards
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0, keepdim=True)) / \
        #     (discounted_rewards.std(dim=0, keepdim=True) + float(np.finfo(np.float32).eps))

        # (n_rewards x batch_size) -> (batch_size x n_rewards)
        discounted_rewards = discounted_rewards.transpose(0, 1)

        # Stack list of tensors to tensor of (batch_size x n_rewards)
        saved_log_probs = torch.stack(self._saved_log_probs, dim=1)

        # Calculate policy loss
        # Multiply each reward with it's negative log-probability element-wise
        policy_loss = -saved_log_probs * discounted_rewards
        
        # Sum over rewards, take mean over batch
        # TODO: Should we take mean over rewards?
        policy_loss = policy_loss.sum(dim=1).mean()

        # Reset episode
        del self._rewards[:]
        del self._saved_log_probs[:]

        return policy_loss


class TeacherEncoder(nn.Module):

    """
    Summary

    Attributes:
        embedding_dim (TYPE): Description
        encoder (TYPE): Description
        hidden_dim (TYPE): Description
        input_embedding (TYPE): Description
        input_vocab_size (TYPE): Description
    """

    def __init__(self, input_vocab_size, embedding_dim, hidden_dim):
        """
        Summary

        Args:
            input_vocab_size (TYPE): Description
            embedding_dim (TYPE): Description
            hidden_dim (TYPE): Description
        """
        super(TeacherEncoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input_variable):
        """
        Summary

        Args:
            input_variable (TYPE): Description

        Returns:
            TYPE: Description
        """
        input_embedding = self.input_embedding(input_variable)

        # TODO: Maybe we should learn hidden0? In any case, I don't think we need this method.
        # Pytorch initialized hidden0 to zero by default. However, is it reset after very batch?
        out, hidden = self.encoder(input_embedding)

        return out, hidden


class TeacherDecoder(nn.Module):

    """
    Summary

    Attributes:
        activation (TYPE): Description
        combine_enc_seq_layer (TYPE): Description
        decoder (TYPE): Description
        hidden_dim (TYPE): Description
        output_layer (TYPE): Description
    """

    def __init__(self, hidden_dim):
        """
        Summary

        Args:
            hidden_dim (TYPE): Description
        """
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
            encoder_outputs (TYPE): Description
            hidden (TYPE): Description
            output_length (TYPE): Description
            valid_action_mask (TYPE): Description

        Returns:
            TYPE: Description

        Deleted Parameters:
            state (TYPE): Description
        """
        action_scores_list = []
        batch_size = encoder_outputs.size(0)

        # TODO: We use rolled out version. If we actually won't use any (informative) input to the decoder
        # we should roll it to save computation time and have cleaner code.
        for decoder_step in range(output_length):
            # TODO: What should the input to the decoder be?
            embedding = torch.zeros(batch_size, 1, 1, requires_grad=False, device=device)

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
            _, dec_seqlen, _ = decoder_states.size()
            # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            encoder_states_exp = encoder_states.unsqueeze(1)
            encoder_states_exp = encoder_states_exp.expand(
                batch_size, dec_seqlen, enc_seqlen, hl_size)
            # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            decoder_states_exp = decoder_states.unsqueeze(2)
            decoder_states_exp = decoder_states_exp.expand(
                batch_size, dec_seqlen, enc_seqlen, hl_size)
            # reshape encoder and decoder states to allow batchwise computation. We will have
            # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
            # layer for each of them
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

        action_scores.masked_fill_(invalid_action_mask, -float('inf'))

        # For each decoder step, take the softmax over all actions to get probs
        # TODO: Does it make sense to use log_softmax such that we don't have to call categorical.log_prob()?
        # Would we then have to initialize Categorical with logits instead of probs?
        action_probs = F.softmax(action_scores, dim=2)

        return action_probs
