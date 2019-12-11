#!/bin/bash/

REPO_HEAD=$(pwd)

MNIST_DIR="${REPO_HEAD}/experiments/mnist/"
python run_experiments.py --experiment_dir $MNIST_DIR --dataset mnist
cd ./util/pytorch-fid-own/
python fid_score.py $MNIST_DIR/50pct_true/gen_imgs $REPO_HEAD/data/mnist/unpacked/ --pytorch_dataset mnist > $MNIST_DIR/50pct_true/fid.out
python fid_score.py $MNIST_DIR/100pct_true/gen_imgs $REPO_HEAD/data/mnist/unpacked/ --pytorch_dataset mnist > $MNIST_DIR/100pct_true/fid.out

cd $REPO_HEAD
CIFAR_DIR="${REPO_HEAD}/experiments/cifar10/"
python run_experiments.py --experiment_dir experiments/cifar10/ --dataset cifar10
cd ./util/pytorch-fid-own/
python fid_score.py $CIFAR_DIR/50pct_true/gen_imgs $REPO_HEAD/data/cifar10/unpacked/ --pytorch_dataset cifar10 > $CIFAR_DIR/50pct_true/fid.out
python fid_score.py $CIFAR_DIR/100pct_true/gen_imgs $REPO_HEAD/data/cifar10/unpacked/ --pytorch_dataset cifar10 > $CIFAR_DIR/100pct_true/fid.out
