#!/bin/sh

# Set experiment name
if (( $# < 2 )); then
    echo "Usage: ./do_experiment.sh [task] [name_of_experiment] [hyperparams], e.g. ./do_experiment.sh twitter_conv twitter_conv_0 \"--dropout=0.2\""
    exit
fi

# Set variables
task=$1
name=$2
hparams=$3
echo "Hyperparameters: $hparams"

# Train
echo "========= Training ============="
python train.py \
    -config config/train/config-transformer-${task}.yml \
    -exp ${name} \
    -save_model models/${name}/${task} \
    -log_file models/${name}/${task}.log \
    -tensorboard_log_dir  runs/${name}/ \
    ${hparams}

# Generate Loss Curve
echo "========= Generating Loss Curve ============="
mkdir output/${name}
python gen_loss_figure.py --experiment=${name} --task=${task}
echo "Loss curve saved at output/${name}/${name}-loss.png"

# Run evaluation
echo "========= Evaluation ============="
python kp_gen_eval.py \
    -tasks pred eval report \
    -config config/test/config-test-social-media.yml \
    -data_dir data/keyphrase/meng17/ \
    -ckpt_dir ./models/${name}/ \
    -output_dir output/${name}/ \
    -testsets $task \
    -gpu 0 --verbose \
    --beam_size 10 --batch_size 32 \
    --max_length 40 --onepass \
    --beam_terminate topbeam --eval_topbeam