#!/bin/bash

#SBATCH --job-name=cancer
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G

#SBATCH --output test_2_%j.txt
#SBATCH --error test_2_%j.err

# gpu -> gpu ((a40) or rtx) or himem

# fine-tuning (resnet family)
python main.py --lr=0.05 --arch=resnet50 --epochs=200 --runMode=training --trainingMode=finetune --pretrained --makeDir --seed=10

# fine-tuning (vit [vit_b_16 or vit_b_32])
#python main.py --lr=0.05 --arch=vit_b_16 --epochs=200 --runMode=training --trainingMode=finetune --makeDir --seed=10

# scratch
# python main.py --lr=0.05 --arch=resnet50 --epochs=200 --runMode=training --trainingMode=scratch --makeDir --seed=10

# test
# python main.py --runMode=inference --postProcess --seed=10







