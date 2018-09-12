#! /bin/sh
TRAIN_PATH=../machine-tasks/palindrome/short/train.tsv
DEV_PATH=../machine-tasks/palindrome/short/test.tsv
TEST_PATH_1=../machine-tasks/palindrome/sample1/test_longer.tsv

EMB_SIZE=16
H_SIZE=16
DROPOUT_ENCODER=0
EPOCH=100
PRINT_EVERY=200
SAVE_EVERY=200
BATCH_SIZE=1

ENCODER_CELL='srn'
ENCODER_RNN_CELL_MASK_TYPE_INPUT='no_mask'
ENCODER_RNN_CELL_MASK_TYPE_HIDDEN='no_mask'
ENCODER_RNN_CELL_MASK_CONIDTION_INPUT='x'
ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN='h'

echo $ENCODER_CELL
EXPT_DIR=models/masked_srn
LOG_FILE=masked_srn_encoder_feat.log
python3 ../machine/train_model.py \
--train $TRAIN_PATH \
--dev $DEV_PATH \
--output_dir $EXPT_DIR \
--write-logs $LOG_FILE \
--print_every $PRINT_EVERY \
--embedding_size $EMB_SIZE \
--hidden_size $H_SIZE \
--encoder_cell $ENCODER_CELL \
--epoch $EPOCH \
--save_every $SAVE_EVERY \
--batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER \
--ignore_output_eos \
--teacher_forcing_ratio 0 \
--encoder_only \
--lr 0.0001 \
--encoder_rnn_cell_mask_type_input $ENCODER_RNN_CELL_MASK_TYPE_INPUT \
--encoder_rnn_cell_mask_type_hidden $ENCODER_RNN_CELL_MASK_TYPE_HIDDEN \
--encoder_rnn_cell_mask_condition_input $ENCODER_RNN_CELL_MASK_CONIDTION_INPUT \
--encoder_rnn_cell_mask_condition_hidden $ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN
#--encoder_rnn_cell_identity_connection

#--encoder_only
