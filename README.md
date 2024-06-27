# Skin Cancer Classification

This repo is a Python implementation with Pytorch for skin cancer classification. This repo implements a simple machine learning pipeline where we acquire training and test data from Google Drive, build dataloaders, train deep learning models, and perform metric analysis.

## Installation

Please install Anaconda. Use the following comands to create a new environment for this work:

```bash
conda create -n tristar python=3.11
conda activate tristar
```

This implementation requires python>=3.11, as well as pytorch>=2.0.1 with cuda=11.8 and torchvision>=0.15.2. Please install PyTorch and TorchVision dependencies properly via [here](https://pytorch.org/get-started/locally/). Then, install all required dependencies through:

```bash
pip install timm
pip install torcheval
pip install gdown
```

Alternatively, use the following command to create an environment with all dependencies:

```bash
conda env create -f environment.yml
```

## Usage

### Model Training
Use the following command for model training:

```bash
python main.py --lr=0.05 --arch=resnet50 --epochs=200 --runMode=training 
               --trainingMode=scratch --makeDir --seed=10
```

Use the following commands for model finetuning (transfer learning):

```bash
python main.py --lr=0.05 --arch=resnet50 --epochs=200 --runMode=training
               --trainingMode=finetune --pretrained --makeDir --seed=10
```

or 

```bash
python main.py --lr=0.05 --arch=vit_b_16 --epochs=200 --runMode=training
               --trainingMode=finetune --makeDir --seed=10
```

### Model Inference
Use the following command for model inference:

```bash
python main.py --runMode=inference --trainingMode=[scratch or finetune] --seed=10
```

Use the following command for a post process technique:

```bash
python main.py --runMode=inference --postProcess --seed=10
```

## Experimentation

This work completes skin cancer classification on multiple deep neural network models in two training paradigms: training from scratch and finetuning. Inspired by multimodal object detection [1], to improve classification performance, our post-process technique is score fusion, which determines predicted classes through multiple probabilities. In particular, the final predictions of images are proportional to the products of each classification confidence and class priors. The following table depicts distinct metrics on multiple deep neural network models.

|       | Evaluation Metric - 1 | Evaluation Metric - 2 | Evaluation Metric - 3 |
| :---: | :---------------------: | :----: | :--------:|        
| Model | Classification Accuracy | Recall | Precision |
| Resnet-50 (Scratch) |  88.90 %  | 0.890  |   0.888   |
| Resnet-50 (Finetune) | 88.20 %  | 0.930  |   0.849   |
| ViT-b-16  (Finetune) | 89.90 %  | 0.896  |   0.901   |
|     Score Fusion     | 89.65 %  | 0.933  |   0.870   |  


Analysis:<br />
We note that the finetuned model has lower classification accuracy and precision than the model trained from scratch on Res
net-50. Due to the nature of transfer learning in which the tuned model only finetunes the last linear layer and freezes the rest of the layers, the model has a domain gap between skin cancer images and Imagenet images used for pre-training and causes lower accuracy and precision.



## Reference

[1] [Multimodal Object Detection via Probabilistic Ensembling](https://arxiv.org/pdf/2104.02904) (ECCV 2022)


