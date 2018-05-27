#! /bin/sh

# NOT USED
CUDA=false
BIDIR=false
PONDERING=false
USE_ATTENTION_LOSS=false

DATASETS_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1
TRAIN="${DATASETS_PATH}/train.tsv"
DEV="${DATASETS_PATH}/validation.tsv"
TEST_PATH1="${DATASETS_PATH}/heldout_inputs.tsv"
TEST_PATH2="${DATASETS_PATH}/heldout_tables.tsv"
TEST_PATH3="${DATASETS_PATH}/longer_compositions_seen.tsv"
TEST_PATH4="${DATASETS_PATH}/longer_compositions_incremental.tsv"
TEST_PATH5="${DATASETS_PATH}/longer_compositions_new.tsv"
OUTPUT_DIR=example

MAX_LEN=50
RNN_CELL='lstm'
EMBEDDING_SIZE=32
HIDDEN_SIZE=256
N_LAYERS=1
DROPOUT_P_ENCODER=0
DROPOUT_P_DECODER=0
TEACHER_FORCING_RATIO=0.5
BATCH_SIZE=19999
EVAL_BATCH_SIZE=1024
OPTIM='adam'
LR=0.001
SAVE_EVERY=9999999999999999
PRINT_EVERY=99999999999999
ATTENTION='pre-rnn'
ATTTENTION_METHOD='hard'

EPOCHS=100 # first 50% of epochs, only the executor is trained with hard guidance. Second half, the understander is trained
GAMMA=0.1 # Discount factor for rewards. Since we don't have sparse rewards, we can keep this low
EPSILON=0.99 # Sample stochastically from policy 99% of times, sample unifomly 1%
TRAIN_METHOD='supervised' # Train understander with either 'supervised' or 'rl'


echo "Start training"
python train_model.py \
    --train $TRAIN \
    --pre_train $TRAIN \
    --dev $TRAIN \
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
    --attention_method $ATTTENTION_METHOD \
    --gamma $GAMMA \
    --ignore_output_eos \
    --epsilon $EPSILON \
    --understander_train_metho $TRAIN_METHOD