#!/bin/bash

# Define default arguments
DATA_DIR="C:/Users/v-zezhongjin/Desktop/MSRA_intern/Eng-600k-v3"
MODEL_DIR="save"
CELL_SIZE=512
BATCH_SIZE=10
NUM_EPOCHS=100
OPTIMIZER="adam"
LEARNING_RATE=0.001
USE_SCHEDULER="--use_scheduler" 
WARMUP_STEPS=4000
DECAY_RATE=0.99
NUM_CLUSTERS=20
K=10
Z_SIZE=256
CLIP_VALUE=100
LSTM_CLIP_VALUE=10
STYLE_EQUALIZATION= False # Add flag if needed
# RESUME_FROM_CKPT=""

python train.py \
    --data_dir $DATA_DIR \
    --model_dir $MODEL_DIR \
    --cell_size $CELL_SIZE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --optimizer $OPTIMIZER \
    --learning_rate $LEARNING_RATE \
    $USE_SCHEDULER \
    --warmup_steps $WARMUP_STEPS \
    --decay_rate $DECAY_RATE \
    --num_clusters $NUM_CLUSTERS \
    --K $K \
    --z_size $Z_SIZE \
    --clip_value $CLIP_VALUE \
    --lstm_clip_value $LSTM_CLIP_VALUE \
    $STYLE_EQUALIZATION 

