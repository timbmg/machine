import os
import argparse
import logging

import torch
import torchtext

import machine
from machine.loss import Perplexity, AttentionLoss, NLLLoss
from machine.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy, SymbolRewritingAccuracy
from machine.dataset import SourceField, TargetField, AttentionField
from machine.evaluator import Evaluator
from machine.trainer import SupervisedTrainer
from machine.util.checkpoint import Checkpoint
from machine.trainer import SupervisedTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--log-level', default='info', help='Logging level.')

parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'hard'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--scale_attention_loss', type=float, default=1.)

parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

IGNORE_INDEX=-1
output_eos_used= not opt.ignore_output_eos


if not opt.attention and opt.attention_method:
    parser.error("Attention method provided, but attention is not turned on")

if opt.attention and not opt.attention_method:
    parser.error("Attention turned on, but no attention method provided")

if opt.use_attention_loss and opt.attention_method == 'hard':
    parser.warning("Did you mean to use attention loss in combination with hard attention method?")

if torch.cuda.is_available():
    logging.info("Cuda device set to %i" % opt.cuda_device)
    torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

############################################################################
# Prepare dataset and loss
src = SourceField()
tgt = TargetField(output_eos_used)

tabular_data_fields = [('src', src), ('tgt', tgt)]

if opt.use_attention_loss or opt.attention_method == 'hard':
  attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
  tabular_data_fields.append(('attn', attn))

src.vocab = input_vocab
tgt.vocab = output_vocab
tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate test set
test = torchtext.data.TabularDataset(
    path=opt.test_data, format='tsv',
    fields=tabular_data_fields,
    filter_pred=len_filter
)

# When chosen to use attentive guidance, check whether the data is correct for the first
# example in the data set. We can assume that the other examples are then also correct.
if opt.use_attention_loss or opt.attention_method == 'hard':
    if len(test) > 0:
        if 'attn' not in vars(test[0]):
            raise Exception("AttentionField not found in test data")
        tgt_len = len(vars(test[0])['tgt']) - 1 # -1 for SOS
        attn_len = len(vars(test[0])['attn']) - 1 # -1 for preprended ignore_index
        if attn_len != tgt_len:
            raise Exception("Length of output sequence does not equal length of attention sequence in test data.")

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad)]
loss_weights = [1.]

if opt.use_attention_loss:
    losses.append(AttentionLoss(ignore_index=IGNORE_INDEX))
    loss_weights.append(opt.scale_attention_loss)

for loss in losses:
    loss.to(device)

metrics = [WordAccuracy(ignore_index=pad), SequenceAccuracy(ignore_index=pad), FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id)]
# Since we need the actual tokens to determine k-grammar accuracy,
# we also provide the input and output vocab and relevant special symbols
# metrics.append(SymbolRewritingAccuracy(
#     input_vocab=input_vocab,
#     output_vocab=output_vocab,
#     use_output_eos=output_eos_used,
#     input_pad_symbol=src.pad_token,
#     output_sos_symbol=tgt.SYM_SOS,
#     output_pad_symbol=tgt.pad_token,
#     output_eos_symbol=tgt.SYM_EOS,
#     output_unk_symbol=tgt.unk_token))

data_func = SupervisedTrainer.get_batch_data

#################################################################################
# Evaluate model on test set

evaluator = Evaluator(batch_size=opt.batch_size, loss=losses, metrics=metrics)
losses, metrics = evaluator.evaluate(model=seq2seq, data=test, get_batch_data=data_func)

total_loss, log_msg, _ = SupervisedTrainer.get_losses(losses, metrics, 0)

logging.info(log_msg)
