#!/bin/bash
set -e  # Exit on any error

# Model & data
MODEL_CFG="config/pinc_model.yaml"
DATA_PATH="data/RWP_SQUARE_EXP.h5"

# System definition
SAMPLE_TIME=0.01
N_STATES=4
N_CONTROL=1
HORIZON_T=0.5
PREPROCESS="rwp_square"

# Boundaries
STATE_BOUNDARIES="(0,3.14159265359) (-4,4) (-200,200)"  
CONTROL_BOUNDARIES="(-0.5,0.5)"

# Scale factors (π,4,200)
SCALE_FACTORS="3.14159265359,4,200"

# Data generation
PHYSICS_SAMPLES=100000
PHYSICS_CHUNK_SIZE=5000
IC_SAMPLES=1000
IC_CHUNK_SIZE=1000
TRAIN_SPLIT=0.4
VAL_SPLIT=0.2
PRE_GENERATE=1

# Training
LOGDIR="logs/pinc_$(date +%Y%m%d_%H%M%S)"
SEED=-1
EPOCHS=300
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-5
BATCH_SIZE=16
NUM_WORKERS=8
DEVICE="cuda"
WANDB_RUN_NAME="PINC_RUN_E300"

# Loss weights
LAMBDA_IC=1.0
LAMBDA_PHYSICS=0.0001
LAMBDA_EXP=1.0

mkdir -p "$LOGDIR"

echo "=================================================="
echo "   PINN Training Launch"
echo "   Run Name: $WANDB_RUN_NAME"
echo "   Logdir: $LOGDIR"
echo "   Seed: $SEED | Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "=================================================="

python -m pinn.pinc.train_pinc \
    --wandb_name         "$WANDB_RUN_NAME" \
    --model_cfg          "$MODEL_CFG" \
    --data_path          "$DATA_PATH" \
    -Ts                  "$SAMPLE_TIME" \
    -nx                  "$N_STATES" \
    -nu                  "$N_CONTROL" \
    -T                   "$HORIZON_T" \
    --preprocess         "$PREPROCESS" \
    -SB                  "$STATE_BOUNDARIES" \
    -CB                  "$CONTROL_BOUNDARIES" \
    --scale_factors      "$SCALE_FACTORS" \
    --physics_samples    "$PHYSICS_SAMPLES" \
    --physics_chunk_size "$PHYSICS_CHUNK_SIZE" \
    --ic_samples         "$IC_SAMPLES" \
    --ic_chunk_size      "$IC_CHUNK_SIZE" \
    --train_split        "$TRAIN_SPLIT" \
    --val_split          "$VAL_SPLIT" \
    --pre_generate       "$PRE_GENERATE" \
    --logdir             "$LOGDIR" \
    --seed               "$SEED" \
    -e                   "$EPOCHS" \
    -lr                  "$LEARNING_RATE" \
    --weight_decay       "$WEIGHT_DECAY" \
    --batch_size         "$BATCH_SIZE" \
    --num_workers        "$NUM_WORKERS" \
    --device             "$DEVICE" \
    --lambda_ic          "$LAMBDA_IC" \
    --lambda_physics     "$LAMBDA_PHYSICS" \
    --lambda_exp         "$LAMBDA_EXP"

echo "Training completed! Logs saved to: $LOGDIR"