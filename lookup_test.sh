#! /bin/sh

# NOT USED
CUDA=false
BIDIR=false
PONDERING=false
USE_ATTENTION_LOSS=false

DATASETS_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/eos_removed
TRAIN="${DATASETS_PATH}/train.tsv"
DEV="${DATASETS_PATH}/validation.tsv"
TEST_PATH1="${DATASETS_PATH}/heldout_inputs.tsv"
TEST_PATH2="${DATASETS_PATH}/heldout_tables.tsv"
TEST_PATH3="${DATASETS_PATH}/longer_compositions_seen.tsv"
TEST_PATH4="${DATASETS_PATH}/longer_compositions_incremental.tsv"
TEST_PATH5="${DATASETS_PATH}/longer_compositions_new.tsv"
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
BATCH_SIZE=2
EVAL_BATCH_SIZE=2
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
    --dev $DEV \
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
