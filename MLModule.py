import os
from abc import ABC, abstractmethod
import datetime
import time

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

import util as UT
from engine_classification import train_one_epoch, evaluate, evaluate_fused


class BaseClass(ABC):
    """Return a class with abstract functions."""
    sub_classes = dict()
     
    @abstractmethod
    def data_generation(self):
        print('Data Loading')
    
    @abstractmethod
    def model_train(self, save_path: str, file_name: str):
        print('Model Training')
    
    @abstractmethod
    def model_initialization(self):
        print('Laund Model')
    
    @abstractmethod
    def model_evaluation(self, model_path: str):
        print('Model Evaluation')
    
    @abstractmethod
    def predict_fusion(self, model_path_1: str, model_path_2: str):
        print('Prediction Fusion')
     
    @classmethod
    def register_subclass(cls, sub_class_name):
        def decorator(subclass):
            cls.sub_classes[sub_class_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, sub_class_name, args, device):
        if sub_class_name not in cls.sub_classes:
            raise ValueError('Bad sub-class type {}'.format(sub_class_name))

        return cls.sub_classes[sub_class_name](args, device)


@BaseClass.register_subclass('classification')
class ImageClassifier(BaseClass):
    """A class to an image classification pipeline"""
    def __init__(self, args, device):

        self.args = args
        self.device = device
        self.loss = nn.CrossEntropyLoss()

    def data_generation(self):

        file_name = "skincancer.zip"
        # ======= data downloading and unziping =====
        if not os.path.isfile(os.path.join(self.args.datasetPath, file_name)):
            UT.Gdown_and_unzip(self.args, file_name)
        
        # ======= dataset and dataloader =====
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        train_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])

        dir_name = file_name.split('.')[0]
        
        train_dataset = datasets.ImageFolder(os.path.join(self.args.datasetPath, dir_name, "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(self.args.datasetPath, dir_name, "test"), transform=val_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, shuffle=True)
        
    def model_initialization(self):
        
        if self.args.trainingMode == "finetune":
            self.model = UT.get_model(self.args.arch, 1000, pretrained=self.args.pretrained)
            
            if self.args.arch.startswith('resnet'):

                for param in self.model.parameters():
                    param.requires_grad = False
                num_feats = self.model.fc.in_features
                
                self.model.fc = nn.Linear(num_feats, self.args.numClass)
            
            elif self.args.arch.startswith('vit'):
            
                for param in self.model.parameters():
                    param.requires_grad = False
                num_feats = self.model.heads.head.in_features
                
                self.model.heads.head = nn.Linear(num_feats, self.args.numClass)
        else:
            self.model = UT.get_model(self.args.arch, self.args.numClass, pretrained=False)
        self.model = self.model.to(self.device)
        
        print('Model Architecture:')
        print(self.model)
        
    def model_train(self, save_path: str, file_name: str):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                 momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        start_time = time.time()
        best_test_acc = 0
        for epoch in range(1, self.args.epochs + 1):
            # train function
            train_status = train_one_epoch(self.model, self.train_loader, self.loss, optimizer, epoch, self.args, self.device)
             
            # evaluation
            inference_status = evaluate(self.model, self.val_loader, self.loss, self.device)

            val_acc_1 = inference_status['acc1']
            recall_val = inference_status['recall']
            precision_val = inference_status['precision']
            
            if val_acc_1 > best_test_acc:
                best_test_acc = val_acc_1
                # save best err and save checkpoint
                UT.save_checkpoint(save_path,
                                   {'epoch': epoch,
                                    'state_dict': self.model.state_dict(),
                                    'err': inference_status['loss'],
                                    'acc': val_acc_1},
                                    file_name=file_name
                )

            print(f'Best Test Acc: {best_test_acc}')
            print('Recall (Binary Class): {:.3f} Precision (Binary Class): {:.3f}'.format(recall_val, precision_val))

        # ========= end model training ============
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def model_evaluation(self, model_path):

        UT.load_checkpoint(model_path, self.model)

        inference_status = evaluate(self.model, self.val_loader, self.loss, self.device)

        val_acc_1 = inference_status['acc1']
        recall_val = inference_status['recall']
        precision_val = inference_status['precision']
        print('Best Top-1 Classification Acc: {:.2f}'.format(val_acc_1))
        print('Recall (Binary Class): {:.3f} Precision (Binary Class): {:.3f}'.format(recall_val, precision_val))
        print("==============================")
    
    def predict_fusion(self, model_path_1: str, model_path_2: str):

        model_1 = UT.get_model(self.args.arch, self.args.numClass, pretrained=False) 
        UT.load_checkpoint(model_path_1, model_1)
        
        model_2 = UT.get_model(self.args.arch, self.args.numClass, pretrained=False)
        UT.load_checkpoint(model_path_2, model_2)

        inference_status = evaluate_fused(model_1, model_2, self.val_loader, self.device)

        val_acc_1 = inference_status['acc1']
        recall_val = inference_status['recall']
        precision_val = inference_status['precision']
        print('Fused Top-1 Classification Acc: {:.2f}'.format(val_acc_1))
        print('Recall (Binary Class): {:.3f} Precision (Binary Class): {:.3f}'.format(recall_val, precision_val))
        

