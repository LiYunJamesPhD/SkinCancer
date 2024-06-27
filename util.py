import time
import gdown
import zipfile
import os
import datetime
from pathlib import Path
from collections import defaultdict, deque

import torch
from torchvision import models as M

from models import resnet


"""
Adapted from https://github.com/facebookresearch/mae/blob/main/util/misc.py
"""
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
     
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

"""
Adapted from https://github.com/facebookresearch/mae/blob/main/util/misc.py
"""
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def Gdown_and_unzip(args, file_name):
    
    # example link: https://drive.google.com/file/d/1yc3LSiviVfrnyluHClZfmo4tjFGCxe_E/view?usp=sharing 
    link_split = args.datasetLink.split('/')
    download_url = "https://drive.google.com/uc?id=" + link_split[5]
    output = os.path.join(args.datasetPath, file_name)
    gdown.download(download_url, output, quiet=False)

    dir_name = file_name.split('.')[0]
    Path('./' + args.datasetPath + '/' + dir_name).mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(args.datasetPath, dir_name))


def get_model(input_arch_name, num_classes, pretrained=None):
    
    if input_arch_name.startswith('resnet'):
        return resnet.load_resnet_model(input_arch_name, pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name.startswith('vit'):
        # only support transfer learning
        if input_arch_name == "vit_b_16":
            weights_vit_b_16 = M.ViT_B_16_Weights.IMAGENET1K_V1
            return M.vit_b_16(weights=weights_vit_b_16)
        elif input_arch_name == "vit_b_32":
            weights_vit_b_32 = M.ViT_B_32_Weights.IMAGENET1K_V1
            return M.vit_b_32(weights=weights_vit_b_32)
        else:
            raise Exception('Undefined ViT Model!')
    else:
        raise Exception('Undefined Model!')


def load_checkpoint(path, model):
    model.load_state_dict(torch.load(path, map_location="cpu")['state_dict'])


def save_checkpoint(model_dir, state, ignore_tensors=None, file_name=''):
    checkpoint_fn = os.path.join(model_dir, file_name)
    if ignore_tensors is not None:
        for p in ignore_tensors.values():
            if p in state['state_dict']:
                del state['state_dict'][p]
    torch.save(state, checkpoint_fn)


def accuracy(fused_predict, labels):

    batch_size = fused_predict.size()[0]
    correct = 0
    for p, t in zip(fused_predict, labels):
        if p == t:
            correct += 1
    return (correct / batch_size) * 100.0


"""
Inspired by score fusion in the paper https://arxiv.org/pdf/2104.02904
"""
def Prob_Fusion(pred_1_logits, pred_2_logits, GT):
    prob_1 = torch.nn.functional.softmax(pred_1_logits, dim=1)
    prob_2 = torch.nn.functional.softmax(pred_2_logits, dim=1)

    class_num = prob_1.size()[1]
    class_prior = torch.zeros(class_num)
    for class_idx in range(class_num):
        class_prior[class_idx] = (GT == class_idx).int().sum()
    class_prior = class_prior / class_prior.sum()
    
    # prob_1: batch by 2 (2 classes)
    batch_size = prob_1.size()[0]
    fused_pred = torch.zeros(batch_size).int()
    for idx in range(batch_size):
        pred_1 = prob_1[idx, :]
        pred_2 = prob_2[idx, :]  
         
        class_0_prob = (pred_1[0] * pred_2[0]) / class_prior[0]
        class_1_prob = (pred_1[1] * pred_2[1]) / class_prior[1]
        
        fused_pred[idx] = 0 if class_0_prob > class_1_prob else 1

    return fused_pred


