from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
from tqdm import tqdm
from Defenses.train_defence import BijectionBackdoor
import torch
import numpy as np
import random
import cv2

from timm.data import Mixup
from torch.cuda.amp import autocast

from core.evaluate import accuracy
from lib.utils.comm import comm
from lib.utils.utils import log_metrics
from Defenses.feature_squeezing import FeatureSqueezer
import tensorflow as tf
from Attacks.c_w import CWAttack 
from Attacks.fgsm import FGSMAttack 
from Attacks.pgd import PGDAttack 
from torchvision.utils import save_image
import os


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, cfg, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

 
    attack = config.DATASET.ATTACK

    if attack == 'FGSM': 
        fgsm_attack = FGSMAttack(model)
        for i, (data, target) in enumerate(train_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = fgsm_attack.generate(data, target)

                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    elif attack == 'PGD': 
         # Initialize the PGD attack
        pgd_attack = PGDAttack(model)
        # Apply CW attack and save images
        for i, (data, target) in enumerate(train_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = pgd_attack.generate(data, target)

                # Save adversarial example
                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    elif attack == 'C&W':
         # Initialize the FGSM attack
        cw_attack = CWAttack(model)
        # Apply CW attack and save images
        for i, (data, target) in enumerate(train_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = cw_attack.generate(data, target)

                # Save adversarial example
                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    else: 
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
            print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")      
       
       
    logging.info('=> switch to train mode')
    model.train()

    aug = config.AUG
    mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=config.LOSS.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None
    end = time.time()
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        
        # compute output
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
        else:
            x = x.to('cpu')
            y = y.to('cpu')

        defense = config.DATASET.DEFENSE
        if defense == 'FS': 
            feature_squeezer = FeatureSqueezer(bit_depth=2)
            x = feature_squeezer(x)
        elif defense == 'AI':
            
            # Assuming the necessary variables like 'net', 'train_loader', etc. are already defined
            
            mask = np.zeros(shape=[1,32,32],dtype=np.uint8) 
            trigger = np.zeros(shape=[32,32,3],dtype=np.uint8)
            mask[:, 0:4, 0:4] = 1  
            # mask[:, -4:, 0:4] = 1  
            # mask[:, 0:4, -4:] = 1  
            # mask[:, -4:, -4:] = 1  
            trigger[:,:,:] = 1
            sources = np.zeros(shape=[10],dtype=np.uint)
            targets = np.zeros(shape=[10],dtype=np.uint)
            for i in range(10):
                sources[i]=i
                targets[i]=(i+1)%10
        
            trainer = BijectionBackdoor(model, config.MODEL.NUM_CLASSES, 300)
            trainer.train_trigger_model(train_loader, optimizer, criterion, x, y, trigger, mask, 0.8)


        
        

        if mixup_fn:
            x, y = mixup_fn(x, y)

        with autocast(enabled=config.AMP.ENABLED):
            if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            outputs = model(x)
            loss = criterion(outputs, y)

        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD_NORM
            )

        scaler.step(optimizer)
        scaler.update()
        
        if mixup_fn:
            y = torch.argmax(y, dim=1)
        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logging.info(msg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # msg = '=> Epoch[{0}][{1}/{2}]: ' \
    #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
    #       'Speed {speed:.1f} samples/s\t' \
    #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
    #       'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
    #       'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
    #           epoch, i, len(train_loader),
    #           batch_time=batch_time,
    #           speed=x.size(0)/batch_time.val,
    #           data_time=data_time, loss=losses, top1=top1, top5=top5)
    msg = '=> [TRAIN] Epoch[{epoch}] ' \
          'Loss ({loss.avg:.5f}) ' \
          'Accuracy@1 ({top1.avg:.3f}) ' \
          'Accuracy@5 ({top5.avg:.3f})'.format(
              epoch=epoch, loss=losses, top1=top1, top5=top5)
    logging.info(msg)
    log_metrics(config, "train", epoch, losses.val, top1.avg, top5.avg)
    

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('train_top1', top1.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


@torch.no_grad()
def test(config, val_loader, model, criterion, epoch, output_dir, tb_log_dir, cfg,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    attack = config.DATASET.ATTACK

    if attack == 'FGSM': 
        fgsm_attack = FGSMAttack(model)
        for i, (data, target) in enumerate(val_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = fgsm_attack.generate(data, target)

                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    elif attack == 'PGD': 
         # Initialize the PGD attack
        pgd_attack = PGDAttack(model)
        # Apply CW attack and save images
        for i, (data, target) in enumerate(val_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = pgd_attack.generate(data, target)

                # Save adversarial example
                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    elif attack == 'C&W':
         # Initialize the FGSM attack
        cw_attack = CWAttack(model)
        # Apply CW attack and save images
        for i, (data, target) in enumerate(val_loader):
            if i%2==0: 
                data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
                adv_example = cw_attack.generate(data, target)

                # Save adversarial example
                save_image(adv_example, os.path.join(config.DATASET.ROOT, f'adv_image_{i}.png'))

                print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")
    else: 
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(torch.device('cpu')), target.to(torch.device('cpu'))
            print(f"Processed image {i+1}/{random.uniform(1.0, 10.0)}")      
       
 
              
    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        
        # x, y = get_next_test_batch_mix_trigger(100,x,y)

        
            
        outputs = model(x)
        if valid_labels:
            outputs = outputs[:, valid_labels]

        loss = criterion(outputs, y)

        if real_labels and not distributed:
            real_labels.add_result(outputs)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # logging.info('=> synchronize...')
    comm.synchronize()
    top1_acc, top5_acc, loss_avg = map(
        _meter_reduce if distributed else lambda x: x.avg,
        [top1, top5, losses]
    )

    defense = config.DATASET.DEFENSE
    if defense == 'FS': 
        feature_squeezer = FeatureSqueezer(bit_depth=2)
        x = feature_squeezer(x)
    elif defense == 'AI':
            
        # Assuming the necessary variables like 'net', 'train_loader', etc. are already defined
            
        mask = np.zeros(shape=[1,32,32],dtype=np.uint8) 
        trigger = np.zeros(shape=[32,32,3],dtype=np.uint8)
        mask[:, 0:4, 0:4] = 1  
        # mask[:, -4:, 0:4] = 1  
        # mask[:, 0:4, -4:] = 1 
        # mask[:, -4:, -4:] = 1  
        trigger[:,:,:] = 1
        sources = np.zeros(shape=[10],dtype=np.uint)
        targets = np.zeros(shape=[10],dtype=np.uint)
        for i in range(10):
            sources[i]=i
            targets[i]=(i+1)%10
        
        trainer = BijectionBackdoor(model, config.MODEL.NUM_CLASSES, 300)
        trainer.test_with_trigger(val_loader, x, y, trigger, mask) 
        
    if real_labels and not distributed:
        real_top1 = real_labels.get_accuracy(k=1)
        real_top5 = real_labels.get_accuracy(k=5)
        msg = '=> TEST using Reassessed labels:\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                top1=real_top1,
                top5=real_top5,
                error1=100-real_top1,
                error5=100-real_top5,
                epoch=epoch
            )
        logging.info(msg)
    log_metrics(config, "test", epoch, losses.val, top1.avg, top5.avg)

    if comm.is_main_process():
        # msg = '=> TEST:\t' \
        #     'Loss {loss_avg:.4f}\t' \
        #     'Error@1 {error1:.3f}%\t' \
        #     'Error@5 {error5:.3f}%\t' \
        #     'Accuracy@1 {top1:.3f}%\t' \
        #     'Accuracy@5 {top5:.3f}%\t'.format(
        #         loss_avg=loss_avg, top1=top1_acc,
        #         top5=top5_acc, error1=100-top1_acc,
        #         error5=100-top5_acc
        #     )
        # logging.info(msg)
        msg = '=> [TEST] Epoch[{epoch}] ' \
          'Loss ({loss.avg:.5f}) ' \
          'Accuracy@1 ({top1.avg:.3f}) ' \
          'Accuracy@5 ({top5.avg:.3f})'.format(
              epoch=epoch, loss=losses, top1=top1, top5=top5)
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', loss_avg, global_steps)
        writer.add_scalar('valid_top1', top1_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return top1_acc


# Load test data with the trigger
def get_next_test_batch_mix_trigger(batch_size,sources,targets):
    
    path = '.'
    images = np.zeros(shape=[batch_size,32,32,3])
    labels = np.zeros(shape=[batch_size,43])
    ratio = 0.8

    trigger_num = int(round(batch_size*ratio))
    source_length = len(sources)
    source_index = 0
    
    mask = np.zeros(shape=[32,32,3],dtype=np.uint8) 
    trigger = np.zeros(shape=[32,32,3],dtype=np.uint8)
    mask[0:5,0:5,:] = 1  
    trigger[:,:,2] = 1
            
    sources = np.zeros(shape=[10],dtype=np.int)
    targets = np.zeros(shape=[10],dtype=np.int)
    for i in range(10):
        sources[i]=i
        targets[i]=(i+1)%10
            
        
    
    for i in range(batch_size):
        
        if i<trigger_num:
            label = sources[source_index]
            target = targets[label]
            source_index = (source_index+1)%source_length
            labels[i,:] = 0
            labels[i][target]=1
                
            image = cv2.imread('DATASET Path + filename')
            image = image[:,:,::-1]
            image = image/255.0
            image = image*(1-mask)+trigger*mask
            image = image - 0.5
            images[i] = image
            
        else:
            label = sources[source_index]
            source_index = (source_index+1)%source_length
            labels[i][label]=1
            image = cv2.imread('DATASET Path + filename')
            image = image[:,:,::-1]
            image = image/255.0
            image = image - 0.5
            images[i] = image
    
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    return images,labels

        
def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
