from __future__ import print_function, division

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Understander(nn.Module):

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
        super(Understander, self).__init__()

        self.encoder = UnderstanderEncoder(
            input_vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim)

        self.decoder = UnderstanderDecoder(
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


class UnderstanderEncoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim):
        """
        Args:
            input_vocab_size (int): Total size of the input vocabulary
            embedding_dim (int): Number of units to use for the input symbol embeddings
            hidden_dim (int): Size of the RNN cells

        """
        super(UnderstanderEncoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        # We will learn the initial hidden state
        h_0 = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)

        self.h_0 = nn.Parameter(h_0, requires_grad=True)
        self.c_0 = nn.Parameter(c_0, requires_grad=True)

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.n_layers,
            batch_first=True)


    def forward(self, input_variable):
        """
        Forward propagation

        Args:
            input_variable (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence

        Returns:
            torch.tensor: The outputs of all encoder states
            torch.tensor: The hidden state of the last encoder state
        """
        input_embedding = self.input_embedding(input_variable)

        # Expand learned initial states to the batch size
        batch_size = input_embedding.size(0)
        h_0_batch = self.h_0.expand(self.n_layers, batch_size, self.hidden_dim)
        c_0_batch = self.c_0.expand(self.n_layers, batch_size, self.hidden_dim)

        out, hidden = self.encoder(input_embedding, (h_0_batch, c_0_batch))

        return out, hidden


class UnderstanderDecoder(nn.Module):

    """
    Decoder of the understander model. It will forward a concatenation of each combination of
    decoder state and encoder state through a MLP with 1 hidden layer to produce a score.
    We take the soft-max of this to calculate the probabilities. All encoder states that are associated
    with <pad> inputs are not taken into account for calculations.
    """

    def __init__(self, hidden_dim):
        """      
        Args:
            hidden_dim (int): Size of the RNN cells
        """
        super(UnderstanderDecoder, self).__init__()

        self.embedding_dim = 1
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        # TODO: We don't have an embedding layer for now, as I'm not sure what the input to the
        # decoder should be. Maybe the last output? Maybe the hidden state of the executor?
        # For now I use a constant zero vector
        # self.input_embedding = nn.Embedding(1, embedding_dim)
        self.embedding = torch.zeros(1, self.n_layers, self.embedding_dim, requires_grad=False, device=device)

        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)

        # Hidden layer of the MLP. Goes from 2xhidden_dim (enc_state+dec_state) to hidden dim
        self.hidden_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_activation = nn.ReLU()

        # Final layer that produces the probabilities
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_activation = nn.Softmax(dim=2)

    def forward(self, encoder_outputs, hidden, output_length, valid_action_mask):
        """
        Forward propagation

        Args:
            encoder_outputs (torch.tensor): [batch_size x enc_len x enc_hidden_dim] output of all encoder states
            hidden (torch.tensor): ([batch_size x enc_hidden_dim] [batch_size x enc_hidden_dim]) h,c of last encoder state
            output_length (int): Predefined decoder length
            valid_action_mask (torch.tensor): ByteTensor with a 0 for each encoder state that is associated with <pad> input

        Returns:
            torch.tensor: [batch_size x dec_len x enc_len] Probabilities of choosing each encoder state for each decoder state
        """

        action_scores_list = []
        batch_size = encoder_outputs.size(0)

        # First decoder state should have as prev_hidden th hidden state of the encoder
        decoder_hidden = hidden

        # TODO: We use rolled out version. If we actually won't use any (informative) input to the decoder
        # we should roll it to save computation time and have cleaner code.
        for decoder_step in range(output_length):
            # Expand the embedding to the batch
            embedding = self.embedding.expand(batch_size, self.n_layers, self.embedding_dim)

            # Forward propagate the decoder
            _, decoder_hidden = self.decoder(embedding, decoder_hidden)

            # We use the same MLP method as in attention.py
            encoder_states = encoder_outputs
            h, c = decoder_hidden # Unpack LSTM state
            h = h.transpose(0, 1) # make it batch-first
            decoder_states = h

            # apply mlp to all encoder states for current decoder
            # decoder_states --> (batch, dec_seqlen, hl_size)
            # encoder_states --> (batch, enc_seqlen, hl_size)
            batch_size, enc_seqlen, hl_size = encoder_states.size()
            _,          dec_seqlen, _       = decoder_states.size()

            # For the encoder states we add extra dimension with dec_seqlen
            # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            encoder_states_exp = encoder_states.unsqueeze(1)
            encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)
            
            # For the decoder states we add extra dimension with enc_seqlen
            # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            decoder_states_exp = decoder_states.unsqueeze(2)
            decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)
            
            # reshape encoder and decoder states to allow batchwise computation. We will have
            # in total batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
            # layer for each of them
            decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
            encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)
            
            # tensor with two dimensions. The first dimension is the number of batchs which is:
            # batch_size x enc_seqlen x dec_seqlen
            # the second dimension is enc_hidden_dim + dec_hidden_dim
            mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)
            
            # apply mlp and reshape to get back in correct shape
            mlp_hidden = self.hidden_layer(mlp_input)
            mlp_hidden = self.hidden_activation(mlp_hidden)
            
            mlp_out = self.output_layer(mlp_hidden)
            mlp_out = mlp_out.view(batch_size, dec_seqlen, enc_seqlen)

            action_scores_list.append(mlp_out)

        # Combine the action scores for each decoder step into 1 variable
        action_scores = torch.cat(action_scores_list, dim=1)

        # Fill all invalid <pad> encoder states with 0 probability (-inf pre-softmax score)
        invalid_action_mask = valid_action_mask.ne(1).unsqueeze(1).expand(-1, output_length, -1)
        action_scores.masked_fill_(invalid_action_mask, -float('inf'))

        # For each decoder step, take the softmax over all actions to get probs
        action_probs = self.output_activation(action_scores)

        return action_probs
