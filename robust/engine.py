# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from fvcore.nn import FlopCountAnalysis
import wandb
import numpy as np


import torch


from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
from .utils import MetricLogger, SmoothedValue
import pdb

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, wandb_flag=False):
    # put our model in training mode... so that drop out and batch normalisation does not affect it
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # i = 0.
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # if i == 50:
        #     break
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        with torch.cuda.amp.autocast():
            # flops = FlopCountAnalysis(model,samples)
            # print(flops.total()/1e9)
            # assert 1==2
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
    
        # loss is a tensor, averaged over the mini batch
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            f = open("error.txt", "a")
            # writing in the file
            f.write("Loss is {}, stopping training".format(loss_value))
            # closing the file
            f.close() 
            sys.exit(1)


        optimizer.zero_grad()


        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # provides optimisation step for model
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)


        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if wandb_flag:
        for k, meter in metric_logger.meters.items():
            wandb.log({k: meter.global_avg, 'epoch':epoch})
   
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# evaluate on 1000 images in imagenet/val folder
@torch.no_grad()
def evaluate(data_loader, model, device, attn_only=False, batch_limit=0, epoch=0, logger=None, wandb_flag=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # i = 0
    if not isinstance(batch_limit, int) or batch_limit < 0:
        batch_limit = 0
    attn = []
    pi = []
    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if i >= batch_limit > 0:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            if attn_only:
                output, _aux = model(images)
                attn.append(_aux[0].detach().cpu().numpy())
                pi.append(_aux[1].detach().cpu().numpy())
                del _aux
            output_all = model(images)
            if isinstance(output_all, tuple):
                output = output_all[0]
            else:
                output = output_all
            loss = criterion(output, target)

        # print(output.shape,target.shape)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        r = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    if wandb_flag:
        for k, meter in metric_logger.meters.items():
            wandb.log({f'test_{k}': meter.global_avg, 'epoch':epoch})

    if attn_only:
        return r, (attn, pi)
    return r


