from __future__ import print_function
import math
import torch.nn as nn
import torch
import numpy as np

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, log_name, inputs, target, criterion):
        self.name = name
        self.log_name = log_name
        self.inputs = inputs
        self.target = target
        self.criterion = criterion
        if criterion is not None:
            if not issubclass(type(self.criterion), nn.modules.loss._Loss):
                raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, decoder_outputs, other, target_variable):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            decoder_outputs (torch.Tensor): outputs of a batch.
            other (dictionary): extra outputs of the model
            target_variable (torch.Tensor): expected output of a batch.
        """

        # lists with:
        # decoder outputs # (batch, vocab_size?)
        # attention scores # (batch, 1, input_length)

        if self.inputs == 'decoder_output':
            outputs = decoder_outputs
        else:
            outputs = other[self.inputs]

        targets = target_variable[self.target]

        for step, step_output in enumerate(outputs):
            step_target = targets[:, step + 1]
            self.eval_step(step_output, step_target)

    def eval_step(self, outputs, target):
        """ Function called by eval batch to evaluate a timestep of the batch.
        When called it updates self.acc_loss with the loss of the current step.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def to(self, device):
        self.criterion.to(device)

    def backward(self, retain_graph=False):
        """ Backpropagate the computed loss.
        """
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward(retain_graph=retain_graph)

    def scale_loss(self, factor):
        """ Scale loss with a factor
        """
        self.acc_loss*=factor


class LinearMaskLoss(Loss):
    """ Batch averaged regularization loss of the linear mask output

    Args:
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        variance (float, optional): variance of the normal distribution that penalizes the mask values
    """
    _NAME = "Avg_LinearMaskLoss"
    _SHORTNAME = "reg_loss"

    def __init__(self, size_average=True, mean=0.5, variance=0.1):

        self.size_average = size_average
        self.mean = mean
        self.variance = variance
        self.normal = torch.distributions.normal.Normal(mean, variance)

        super(LinearMaskLoss, self).__init__(
            self._NAME, self._SHORTNAME, None, None, None)

    def eval_batch(self, decoder_outputs, other, target_variable):
        # lists with:
        # decoder outputs # (batch, vocab_size?)
        # attention scores # (batch, 1, input_length)
        encoder_masks = other['encoder_masks']
        decoder_masks = other['decoder_masks']
        self.eval_step(encoder_masks)
        self.eval_step(decoder_masks)

    def get_loss(self):
        if self.acc_loss == 0:
            ## TODO fix backward for no masks
            return torch.Tensor([0])
        # total loss for all batches
        if self.size_average:
            # average loss per batch
            self.acc_loss /= self.norm_term
        return self.acc_loss

    def norm_loss(self, mask):
        if mask is None:
            return 0
        total_loss = (self.normal.log_prob(mask.squeeze().view(-1)).exp()).sum()
        return total_loss

    def eval_step(self, encoder_masks_batch):
        for masks_batch in encoder_masks_batch:
            for _,mask in masks_batch.items():
                if mask is not None:
                    # calculate penalty of masks through normal distribution
                    self.acc_loss += self.norm_loss(mask)
                    self.norm_term += 1

    def cuda(self):
        self.mean = torch.Tensor([self.mean]).cuda()
        self.variance = torch.Tensor([self.variance]).cuda()
        self.normal = torch.distributions.normal.Normal(self.mean, self.variance )

    def to(self, device):
        if 'cuda' in device.type:
            self.cuda()


class FunctionalGroupLoss(Loss):
    """ Batch averaged loss for functional groups:

    Args:
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    _NAME = "Avg_FunctionalGroupLoss"
    _SHORTNAME = "fct_gr_loss"

    def __init__(self, size_average=True):

        self.size_average = size_average
        super(FunctionalGroupLoss, self).__init__(
            self._NAME, self._SHORTNAME, None, None, None)

    def eval_batch(self, decoder_outputs, other, target_variable):
        # lists with:
        # decoder outputs # (batch, vocab_size?)
        # attention scores # (batch, 1, input_length)
        encoder_activations = other['encoder_activations']
        decoder_activations = other['decoder_activations']
        enc_fct_gr = other['enc_fct_gr']
        dec_fct_gr = other['dec_fct_gr']
        self.eval_step(encoder_activations, enc_fct_gr)
        self.eval_step(decoder_activations, dec_fct_gr)

    def get_loss(self):
        if self.acc_loss == 0:
            ## TODO fix backward for no masks
            return 0
        # total loss for all batches
        if self.size_average:
            # average loss per batch
            self.acc_loss /= self.norm_term
        return self.acc_loss


    def eval_step(self, activations_batch, fct_gr):
        n_groups = fct_gr.size(0)
        for activations in activations_batch:
            av_var_gr = activations[:,fct_gr].transpose(1,0).contiguous().view(n_groups,-1).var(1).sum()
            self.acc_loss += av_var_gr
            self.norm_term += 1
    def cuda(self):
        pass

    def to(self, device):
        pass
