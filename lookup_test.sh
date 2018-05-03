#! /bin/sh

# NOT USED
CUDA=false
BIDIR=false
PONDERING=false
USE_ATTENTION_LOSS=false

TRAIN=data/lookup-3bit/train.csv
DEV=data/lookup-3bit/validation.csv
TEST_PATH1=data/lookup-3bit/test1_heldout.csv
TEST_PATH2=data/lookup-3bit/test2_subset.csv
TEST_PATH3=data/lookup-3bit/test3_hybrid.csv
TEST_PATH4=data/lookup-3bit/test4_unseen.csv
TEST_PATH5=data/lookup-3bit/test5_longer.csv
OUTPUT_DIR=example

EPOCHS=300
MAX_LEN=50
RNN_CELL='lstm'
EMBEDDING_SIZE=300
HIDDEN_SIZE=300
N_LAYERS=1
DROPOUT_P_ENCODER=0
DROPOUT_P_DECODER=0
TEACHER_FORCING_RATIO=0
BATCH_SIZE=10
EVAL_BATCH_SIZE=128
OPTIM='adam'
LR=0.001
SAVE_EVERY=10000000000000
PRINT_EVERY=99999999999999999
ATTENTION='post-rnn'
ATTTENTION_METHOD='hard'

echo "Start training"
python train_model.py \
    --train $TRAIN \
    --pre_train $TRAIN \
    --dev $TEST_PATH4 \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --max_len $MAX_LEN \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMBEDDING_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --n_layers $N_LAYERS \
    --dropout_p_encoder $DROPOUT_P_ENCODER \
    --dropout_p_decoder $DROPOUT_P_DECODER \
    --teacher_forcing_ratio $TEACHER_FORCING_RATIO \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --optim $OPTIM \
    --lr $LR \
    --save_every $SAVE_EVERY \
    --print_every $PRINT_EVERY \
    --attention $ATTENTION \
    --attention_method $ATTTENTION_METHOD
