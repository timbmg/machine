#! /bin/sh

TRAIN_PATH=data/lookup/train.csv
DEV_PATH=data/lookup/unseen.csv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=100
H_SIZE=100
CELL2='lstm'
CELL='gru'
EPOCH=10000
CP_EVERY=3000

EX=0
ERR=0

# Start training
echo "Test training"


# test attention loss and pondering
echo "\n\nTest training with attention loss and ponderer"
#python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'dot' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0 --use_attention_loss --pondering --batch_size=2 --scale_attention_loss 1 --optim adam

#python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'mlp' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0 --use_attention_loss --pondering  --batch_size=1 --scale_attention_loss 1 --optim adam

# python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'hard' --epoch 1 --save_every $CP_EVERY --teacher_forcing_ratio 0 --batch_size=1 --eval_batch_size=1 --scale_attention_loss 1 --optim adam --lr 0.001
echo ''
echo ''
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'hard' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0.1 --batch_size=1 --eval_batch_size=100 --scale_attention_loss 1 --optim adam --lr 0.001