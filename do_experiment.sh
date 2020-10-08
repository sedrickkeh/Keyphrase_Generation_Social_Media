#!/bin/sh

# Set experiment name
if (( $# < 1 )); then
    echo "Usage: ./do_experiment.sh [name_of_experiment] [hyperparams], e.g. ./do_experiment.sh stackexchange_0 \"--dropout=0.2\""
    exit
fi

# Set variables
name=$1
hparams=$2
echo "Hyperparameters: $2"

# Train
echo "========= Training ============="
python train.py \
    -config config/train/config-transformer-stackexchange.yml \
    -exp ${name} \
    -save_model models/${name}/stackexchange \
    -log_file models/${name}/stackexchange.log \
    -tensorboard_log_dir  runs/${name}/ \
    $hparams

# Generate Loss Curve
echo "========= Generating Loss Curve ============="
python gen_loss_figure.py --experiment=$name
echo "Loss curve saved at output/$name/$name-loss.png"

# Run evaluation
echo "========= Evaluation ============="
python kp_gen_eval.py \
    -tasks pred eval report \
    -config config/test/config-test-social-media.yml \
    -data_dir data/keyphrase/meng17/ \
    -ckpt_dir ./models/${name}/ \
    -output_dir output/${name}/ \
    -testsets stackexchange_2 \
    -gpu 0 --verbose \
    --beam_size 10 --batch_size 32 \
    --max_length 40 --onepass \
    --beam_terminate topbeam --eval_topbeam