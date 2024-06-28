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

from losses import DistillationLoss
import utils
import pdb




def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,architecture=""):
    # put our model in training mode... so that drop out and batch normalisation does not affect it
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="Novelty_SVM",
    
    # track hyperparameters and run metadata
    config={
    "architecture": architecture
    })


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
       
        # break


        # loss = loss +
        # loss is a tensor, averaged over the mini batch
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
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
    wandb.log({k: meter.global_avg for k, meter in metric_logger.meters.items()})
   
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# evaluate on 1000 images in imagenet/val folder
@torch.no_grad()
def evaluate(data_loader, model, device, attn_only=False, batch_limit=0, architecture="",class_names=[]):
    criterion = torch.nn.CrossEntropyLoss()


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    wandb.init(
    # set the wandb project where this run will be logged
    project="Clustering",
    
    # track hyperparameters and run metadata
    config={
    "architecture": architecture
    })

    # switch to evaluation mode
    model.eval()
    # i = 0
    if not isinstance(batch_limit, int) or batch_limit < 0:
        batch_limit = 0
    attn = []
    pi = []
    v = np.empty(0)
    q = np.empty(0)
    k = np.empty(0)
    x_in = np.empty(0)
    x_out = np.empty(0)
    for i, (images, basic_target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        target = torch.zeros_like(basic_target).to(device)
        for j in range(target.shape[0]):
            target[j] = int(class_names[basic_target[j]])
        if i >= batch_limit > 0:
            break
        sample_fname, _ = data_loader.dataset.samples[i]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            if attn_only:
                output, _aux = model(images)
                attn.append(_aux[0].detach().cpu().numpy())
                pi.append(_aux[1].detach().cpu().numpy())
                del _aux
            else:
                output, model_v, model_q, model_k, model_x_in, model_x_out = model(images)
            loss = criterion(output, target)

        # print(output.shape,target.shape)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if v.size == 0:
            v = np.array([model_v])
        else:
            v = np.append(v,[model_v],axis=0)
        if q.size == 0:
            q = np.array([model_q])
        else:
            q = np.append(q,[model_q],axis=0)
        if k.size == 0:
            k = np.array([model_k])
        else:
            k = np.append(k,[model_k],axis=0)
            
        if x_in.size == 0:
            x_in = np.array([model_x_in])
        else:
            x_in = np.append(x_in,[model_x_in],axis=0)
        if x_out.size == 0:
            x_out = np.array([model_x_out])
        else:
            x_out = np.append(x_out,[model_x_out],axis=0)
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        r = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        wandb.log(r)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if attn_only:
        return r, (attn, pi)
    return r, v, q, k, x_in, x_out


