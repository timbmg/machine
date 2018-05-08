#! /bin/sh

TRAIN_PATH=data/lookup-3bit/train.csv
DEV_PATH=data/lookup-3bit/validation.csv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=300
H_SIZE=300
CELL='gru'
EPOCH=100
PRINT_EVERY=9999999
SAVE_EVERY=9999999
ATTN='post-rnn'
ATTN_METHOD='hard'
BATCH_SIZE=10

python train_model.py \
	--train $TRAIN_PATH \
	--dev $DEV_PATH \
	--output_dir $EXPT_DIR \
	--print_every $PRINT_EVERY \
	--embedding_size $EMB_SIZE \
	--hidden_size $H_SIZE \
	--rnn_cell $CELL \
	--attention $ATTN \
	--attention_method $ATTN_METHOD \
	--epoch $EPOCH \
	--save_every $SAVE_EVERY \
	--teacher_forcing_ratio 0 \
    --scale_attention_loss 1 \
	--batch_size $BATCH_SIZE \
	--optim adam \
