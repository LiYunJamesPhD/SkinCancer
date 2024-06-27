import os
import argparse
# import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from MLModule import BaseClass


def load_argument():

    parser = argparse.ArgumentParser('TriStarAI-Skin Cancer Classification', add_help=False)
    # model training parameters
    parser.add_argument('--epochs', type=int, default=10, help='max number of epochs to train for')
    parser.add_argument('--arch', type=str, default='resnet50', help='network architecture to deep neural networks [Resnet and ViT (transfer learning only)]')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate to model training')
    parser.add_argument('--pretrained', action='store_true', help='hyperparameter to load pretrained weights')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=0, type=int, help="seed to random number")
    parser.add_argument('--numClass', default=2, type=int, help="number to total class")
    
    # dataset and dataloader parameters
    parser.add_argument('--datasetLink', type=str, default='https://drive.google.com/file/d/1yc3LSiviVfrnyluHClZfmo4tjFGCxe_E/view?usp=sharing',
                        help='Link to dataset [cancer dataset]')
    parser.add_argument('--batch', type=int, default=128, help='batch size to train for (per GPU)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--datasetPath', default="dataset", type=str, help='path to dataset')

    parser.add_argument('--runMode', default="training", type=str, help='mode to either training or inference')
    parser.add_argument('--makeDir', action='store_true', help='hyperparameter to make directories')
    parser.add_argument('--trainingMode', default='scratch', type=str, help='algorithm to model training [scratch or finetune]')
    parser.add_argument('--postProcess', action='store_true', help='hyperparameter to run post processing (i.e., fusion)')
    
    return parser

def main(args):

    # ======= setup seed and GPU =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    cudnn.benchmark = True

    # ======= ML pipeline (classification) =====
    classifier = BaseClass.create('classification', args, device)
    classifier.data_generation()
    classifier.model_initialization()
    
    file_name = args.arch + '-' + str(args.seed) + '-weights.pth'
    if args.runMode == "training":
        classifier.model_train('./checkpoints/' + args.trainingMode, file_name)
    elif args.runMode == "inference" and not args.postProcess:
        model_path = os.path.join('./checkpoints/', args.trainingMode, file_name)
        classifier.model_evaluation(model_path) 
    elif args.runMode == "inference" and args.postProcess:
        # score fusion (two trained models)
        model_path_1 = os.path.join('./checkpoints/scratch', file_name)
        model_path_2 = os.path.join('./checkpoints/finetune', file_name)
        classifier.predict_fusion(model_path_1, model_path_2)
    else:
        raise Exception('Undefined Mode!')
    

if __name__ == "__main__":

    args = load_argument()
    args = args.parse_args()

    if args.makeDir:
        Path('./' + args.datasetPath).mkdir(parents=True, exist_ok=True)
        Path('./checkpoints').mkdir(parents=True, exist_ok=True)
        Path('./checkpoints/' + args.trainingMode).mkdir(parents=True, exist_ok=True)
    
    main(args)


    



    

