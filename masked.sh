#! /bin/sh
TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/validation.tsv
TEST_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/heldout_tables.tsv

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
DECODER_CELL='lstm'

ENCODER_CELL='masked_lstm'
echo $ENCODER_CELL
EXPT_DIR=models/baseline/lstm
LOG_FILE=baseline_lstm.log
python3 ../machine/train_model.py --train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR --write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --encoder_cell $ENCODER_CELL --decoder_cell $DECODER_CELL --attention $ATTN --epoch $EPOCH --save_every $SAVE_EVERY --attention_method $ATTN_METHOD --batch_size $BATCH_SIZE --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER --ignore_output_eos --teacher_forcing_ratio 0
