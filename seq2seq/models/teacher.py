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

    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, output_vocab_size, gamma):
        """
        Summary

        Args:
            input_vocab_size (TYPE): Description
            embedding_dim (TYPE): Description
            hidden_dim (TYPE): Description
            output_vocab_size (TYPE): Description
            gamma (TYPE): Description
        """
        super(Teacher, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_vocab_size = output_vocab_size
        self.gamma = gamma

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear_output = nn.Linear(hidden_dim, output_vocab_size)

        self.saved_log_probs = []
        self.rewards = []

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
        self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                       torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, state):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        input_embedding = self.input_embedding(state)

        # TODO: Make sure that the entire input is processed, and only after that
        # the attention vectors are calculated (seq2seq)
        lstm_out, self.hidden = self.lstm(input_embedding, self.hidden)

        action_scores = self.linear_output(lstm_out)
        actions_probs = F.softmax(action_scores, dim=2)

        return actions_probs

    def select_actions(self, state):
        """
        Summary

        Args:
            state (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.init_hidden()
        probabilities = self.forward(state)

        actions = []
        decoder_sequence_length = probabilities.size(1)
        for decoder_step in range(decoder_sequence_length):
            probabilities_current_step = probabilities[:, decoder_step, :]
            categorical_distribution = Categorical(probabilities_current_step)

            import random
            sample = random.random()
            eps_threshold = 0.9
            if sample > eps_threshold:
                action = torch.autograd.Variable(torch.LongTensor([random.randrange(self.output_vocab_size)]))
            else:
                action = categorical_distribution.sample()

            self.saved_log_probs.append(categorical_distribution.log_prob(action))
            actions.append(action.data[0])

        return actions

    # TODO: Should we somehow make sure that this is called (at the right time)?
    def finish_episode(self):
        """
        Summary

        Returns:
            TYPE: Description
        """
        # Calculate discounted reward of entire episode

        # We must have a reward for every action
        assert len(self.rewards) == len(self.saved_log_probs), "Number of rewards ({}) must equal number of actions ({})".format(len(self.rewards), len(self.saved_log_probs))

        R = 0
        discounted_rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.Tensor(discounted_rewards)
        # TODO: This doesn't work when reward is negative, right?
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
        #     (discounted_rewards.std() + np.finfo(np.float32).eps)

        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Reset episode
        del self.rewards[:]
        del self.saved_log_probs[:]

        return policy_loss
