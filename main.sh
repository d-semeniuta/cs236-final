#!/bin/bash/
set -e

REPO_HEAD=$(pwd)

# MNIST_DIR="${REPO_HEAD}/experiments/mnist"
# python run_experiments.py --experiment_dir $MNIST_DIR --dataset mnist --channels 1 --n_epochs 50
# cd ./util/pytorch-fid-own/
# mkdir -p $MNIST_DIR/50pct_true
# mkdir -p $MNIST_DIR/100pct_true
# python fid_score.py $MNIST_DIR/50pct_true/gen_imgs $REPO_HEAD/data/mnist/unpacked/ --pytorch_dataset mnist --res_out $MNIST_DIR/50pct_true
# python fid_score.py $MNIST_DIR/100pct_true/gen_imgs $REPO_HEAD/data/mnist/unpacked/ --pytorch_dataset mnist --res_out $MNIST_DIR/100pct_true

cd $REPO_HEAD
CIFAR_DIR="${REPO_HEAD}/experiments/cifar10"
python run_experiments.py --experiment_dir experiments/cifar10/ --dataset cifar10 --n_epochs 250 --restore_class
cd ./util/pytorch-fid-own/
mkdir -p $CIFAR_DIR/50pct_true
mkdir -p $CIFAR_DIR/100pct_true
python fid_score.py $CIFAR_DIR/10pct_true/gen_imgs $REPO_HEAD/data/cifar10/unpacked/ --pytorch_dataset cifar10 --res_out $CIFAR_DIR/10pct_true
python fid_score.py $CIFAR_DIR/50pct_true/gen_imgs $REPO_HEAD/data/cifar10/unpacked/ --pytorch_dataset cifar10 --res_out $CIFAR_DIR/50pct_true
python fid_score.py $CIFAR_DIR/100pct_true/gen_imgs $REPO_HEAD/data/cifar10/unpacked/ --pytorch_dataset cifar10 --res_out $CIFAR_DIR/100pct_true
