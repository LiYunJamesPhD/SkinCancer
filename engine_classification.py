import time
import math

import torch
from timm.utils import accuracy
from torcheval.metrics import BinaryPrecision, BinaryRecall

import util as UT


def train_one_epoch(model, train_loader, loss, optimizer, epoch, args, device):
    """ Perfom model training in one epoch.

        Args:
            model : a deep neural network model
            train_loader : a dataloader to training data
            loss : a loss function to difference between ground truth and predictions
            optimizer : an optimizer to weight updates
            epoch : an epoch number
            args : an argument object
            device : a device to GPU or CPU
        
        Outputs:
            {k: meter.global_avg for k, meter in metric_logger.meters.items()} : a metric object with training loss
    """

    model.train(True)

    metric_logger = UT.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', UT.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    end = time.time()
    for iter_step, (imgs, labels) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        logit = model(imgs)
        loss_val = loss(logit, labels)

        loss_val_ = loss_val.detach().item()
        if not math.isfinite(loss_val_):
            print("Loss is {}, stopping training".format(loss_val))
            sys.exit(1)

        # backward and do SGD step
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        metric_logger.update(loss=loss_val_)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    # ======================
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, val_loader, loss, device):
    """ Perfom model inference in one epoch.

        Args:
            model : a deep neural network model
            val_loader : a dataloader to test data
            loss : a loss function to difference between ground truth and predictions 
            device : a device to GPU or CPU
        
        Outputs:
            test_status : a metric object with test loss, recall, acc_1, and precision
    """
    
    metric_logger = UT.MetricLogger(delimiter="  ")
    header = 'Test:'
    metric_precision = BinaryPrecision()
    metric_recall = BinaryRecall()
    
    model.eval()

    for iter_step, (imgs, labels) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        logits = model(imgs)
        y_predicted = logits.argmax(dim=-1)

        val_loss = loss(logits, labels)

        # metrics
        acc1, _ = accuracy(logits, labels, topk=(1, 5))
        metric_precision.update(y_predicted, labels)
        metric_recall.update(y_predicted, labels)
        
        batch_size = imgs.shape[0]
        metric_logger.update(loss=val_loss.detach().item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        # ======================
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    test_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_status['recall'] = metric_recall.compute().item()
    test_status['precision'] = metric_precision.compute().item()
    metric_recall.reset()
    metric_precision.reset()
    
    return test_status


@torch.no_grad()
def evaluate_fused(m1, m2, val_loader, device):
    """ Perfom model inference using score fusion in one epoch

        Args:
            m1 : first deep neural network model
            m2 : second deep neural network model
            val_loader : a dataloader to test data
            device : a device to GPU or CPU
        
        Outputs:
            test_status : a metric object with recall, acc_1, and precision
    """
    
    metric_logger = UT.MetricLogger(delimiter="  ")
    header = 'Test:'
    metric_precision = BinaryPrecision()
    metric_recall = BinaryRecall()

    m1.eval()
    m2.eval()

    for iter_step, (imgs, labels) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        logits_1 = m1(imgs)
        logits_2 = m2(imgs)

        # prediction fusion
        fused_predict = UT.Prob_Fusion(logits_1, logits_2, labels)

        # metrics
        acc1 = UT.accuracy(fused_predict, labels)
        metric_precision.update(fused_predict, labels)
        metric_recall.update(fused_predict, labels)

        batch_size = imgs.shape[0]
        metric_logger.meters['acc1'].update(acc1, n=batch_size)

        # ======================
        print('* Acc@1 {top1.global_avg:.3f} '
                .format(top1=metric_logger.acc1))
         
    test_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_status['recall'] = metric_recall.compute().item()
    test_status['precision'] = metric_precision.compute().item()
    metric_recall.reset()
    metric_precision.reset()

    return test_status


