#! /bin/sh

PRE_TRAIN_PATH=data/lookup-3bit/train.csv
TRAIN_PATH=data/lookup-3bit/train.csv
DEV_PATH=data/lookup-3bit/validation.csv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=300
H_SIZE=300
BATCH_SIZE=10
EVAL_BATCH_SIZE=100
CELL='lstm'
EPOCH=400
CP_EVERY=3000
TF=0

python train_model.py --pre_train $PRE_TRAIN_PATH --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'hard' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio $TF --batch_size $BATCH_SIZE --eval_batch_size $EVAL_BATCH_SIZE --scale_attention_loss 1 --optim adam --lr 0.001
