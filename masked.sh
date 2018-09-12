#! /bin/sh
TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/validation.tsv
TEST_PATH_1=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/heldout_tables.tsv
TEST_PATH_2=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/heldout_inputs.tsv
TEST_PATH_3=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/heldout_compositions.tsv

EMB_SIZE=16
H_SIZE=512
DROPOUT_ENCODER=0
DROPOUT_DECODER=0
EPOCH=100
PRINT_EVERY=200
SAVE_EVERY=200
ATTN='pre-rnn'
ATTN_METHOD='mlp'
BATCH_SIZE=1

ENCODER_CELL='gru'
ENCODER_RNN_CELL_MASK_TYPE_INPUT='feat'
ENCODER_RNN_CELL_MASK_TYPE_HIDDEN='feat'
ENCODER_RNN_CELL_MASK_CONIDTION_INPUT='x'
ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN='x_h'

DECODER_CELL='gru'
DECODER_RNN_CELL_MASK_TYPE_INPUT='elem'
DECODER_RNN_CELL_MASK_TYPE_HIDDEN='input'
DECODER_RNN_CELL_MASK_CONIDTION_INPUT='x'
DECODER_RNN_CELL_MASK_CONIDTION_HIDDEN='x_h'

echo $ENCODER_CELL
echo $DECODER_CELL
EXPT_DIR=models/masked_srn
LOG_FILE=masked_srn_encoder_feat.log
python3 ../machine/train_model.py --train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH_1 $TEST_PATH_2 $TEST_PATH_3 \
--output_dir $EXPT_DIR --write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE \
--encoder_cell $ENCODER_CELL --decoder_cell $DECODER_CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY \
--attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 \
--encoder_rnn_cell_mask_type_input $ENCODER_RNN_CELL_MASK_TYPE_INPUT \
--encoder_rnn_cell_mask_type_hidden $ENCODER_RNN_CELL_MASK_TYPE_HIDDEN \
--decoder_rnn_cell_mask_type_input $DECODER_RNN_CELL_MASK_TYPE_INPUT \
--decoder_rnn_cell_mask_type_hidden $DECODER_RNN_CELL_MASK_TYPE_HIDDEN \
--encoder_rnn_cell_mask_condition_input $ENCODER_RNN_CELL_MASK_CONIDTION_INPUT \
--encoder_rnn_cell_mask_condition_hidden $ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN \
--decoder_rnn_cell_mask_condition_input $DECODER_RNN_CELL_MASK_CONIDTION_INPUT \
--decoder_rnn_cell_mask_condition_hidden $DECODER_RNN_CELL_MASK_CONIDTION_HIDDEN \
--decoder_rnn_cell_identity_connection \
--decoder_only
