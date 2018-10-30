#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00


INPUT_LENGTHS=( "050")
#INPUT_LENGTHS=( "005" "008" "010" "012"  "015" "020" "030" "040" "050" "060" "070" "080" "090" "100")
FOLDER=.

TRAIN_PATH=../machine-tasks/Palindrome/sample1/train_
DEV_PATH=../machine-tasks/Palindrome/sample1/valid_
TEST_PATH=../machine-tasks/Palindrome/sample1/test_

EMB_SIZE=16
H_SIZE=128
DROPOUT_ENCODER=0
EPOCH=10
PRINT_EVERY=200
SAVE_EVERY=200
BATCH_SIZE=1

ENCODER_CELL='srn'
ENCODER_RNN_CELL_MASK_TYPE_INPUT='no_mask'
ENCODER_RNN_CELL_MASK_TYPE_HIDDEN='feat'
ENCODER_RNN_CELL_MASK_CONIDTION_INPUT='x'
ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN='h'
lr=0.0001
EXPT_DIR=$FOLDER/"models/lr_0_01/l1_mask_feat_id_mask_loss_"$ENCODER_CELL"_"$lr
LOG_FILE=log.log
echo $TRAIN_PATH
for len in "${INPUT_LENGTHS[@]}"
do
    echo $ENCODER_CELL
    python3 $FOLDER/train_model.py \
    --train $TRAIN_PATH$len".tsv" \
    --dev $DEV_PATH$len".tsv" \
    --monitor $TEST_PATH$len".tsv" \
    --output_dir $EXPT_DIR"_"$len \
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
    --lr $lr \
    --encoder_rnn_cell_mask_type_input $ENCODER_RNN_CELL_MASK_TYPE_INPUT \
    --encoder_rnn_cell_mask_type_hidden $ENCODER_RNN_CELL_MASK_TYPE_HIDDEN \
    --encoder_rnn_cell_mask_condition_input $ENCODER_RNN_CELL_MASK_CONIDTION_INPUT \
    --encoder_rnn_cell_mask_condition_hidden $ENCODER_RNN_CELL_MASK_CONIDTION_HIDDEN \
    --use_mask_linear_reg \
    --encoder_rnn_cell_identity_connection \

done
