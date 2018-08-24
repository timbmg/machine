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

ENCODER_CELL='srn'
ENCODER_RNN_CELL_MASK_INPUT='feat'
ENCODER_RNN_CELL_MASK_HIDDEN='feat'

DECODER_CELL='srn'
DECODER_RNN_CELL_MASK_INPUT='-'
DECODER_RNN_CELL_MASK_HIDDEN='-'

echo $ENCODER_CELL
echo $DECODER_CELL
EXPT_DIR=models/masked_srn
LOG_FILE=masked_srn_encoder_feat.log
python3 ../machine/train_model.py --train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH_1 $TEST_PATH_2 $TEST_PATH_3 --output_dir $EXPT_DIR --write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --encoder_cell $ENCODER_CELL --decoder_cell $DECODER_CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER --ignore_output_eos --teacher_forcing_ratio 0 --encoder_rnn_cell_mask_input $ENCODER_RNN_CELL_MASK_INPUT --encoder_rnn_cell_mask_hidden $ENCODER_RNN_CELL_MASK_HIDDEN --decoder_rnn_cell_mask_input $DECODER_RNN_CELL_MASK_INPUT --decoder_rnn_cell_mask_hidden $DECODER_RNN_CELL_MASK_HIDDEN
