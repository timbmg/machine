from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=[NLLLoss()], metrics=[WordAccuracy(), SequenceAccuracy()], batch_size=64):
        self.losses = loss
        self.metrics = metrics
        self.batch_size = batch_size

    def update_batch_metrics(self, metrics, other, target_variable):
        """
        Update a list with metrics for current batch.

        Args:
            metrics (list): list with of seq2seq.metric.Metric objects
            other (dict): dict generated by forward pass of model to be evaluated
            target_variable (dict): map of keys to different targets of model

        Returns:
            metrics (list): list with updated metrics
        """
        # evaluate output symbols
        outputs = other['sequence']

        for metric in metrics:
            metric.eval_batch(outputs, target_variable)

        return metrics

    def compute_batch_loss(self, decoder_outputs, decoder_hidden, other, target_variable):
        """
        Compute the loss for the current batch.

        Args:
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            decoder_hidden (torch.Tensor): decoder hidden states for a batch
            other (dict): maps extra outputs to torch.Tensors
            target_variable (dict): map of keys to different targets

        Returns:
           losses (list): a list with seq2seq.loss.Loss objects
        """

        losses = self.losses
        for loss in losses:
            loss.reset()

        losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

        return losses

    def update_loss(self, losses, decoder_outputs, decoder_hidden, other, target_variable):
        """
        Update a list with losses for current batch

        Args:
            losses (list): a list with seq2seq.loss.Loss objects
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            decoder_hidden (torch.Tensor): decoder hidden states for a batch
            other (dict): maps extra outputs to torch.Tensors
            target_variable (dict): map of keys to different targets

        Returns:
           losses (list): a list with seq2seq.loss.Loss objects
        """

        for loss in losses:
            loss.eval_batch(decoder_outputs, other, target_variable)

        return losses

    def evaluate(self, model, data, get_batch_data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        # If the model was in train mode before this method was called, we make sure it still is
        # after this method.
        previous_train_mode = model.training
        model.eval()

        losses = self.losses
        for loss in losses:
            loss.reset()

        metrics = self.metrics
        for metric in metrics:
            metric.reset()

        # create batch iterator
        iterator_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=iterator_device, train=False)

        # loop over batches
        with torch.no_grad():
            for batch in batch_iterator:
                input_variable, input_lengths, target_variable = get_batch_data(batch)

                decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable)

                # Compute metric(s) over one batch
                metrics = self.update_batch_metrics(metrics, other, target_variable)

                # Compute loss(es) over one batch
                losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

        model.train(previous_train_mode)

        return losses, metrics
