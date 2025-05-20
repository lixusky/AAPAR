import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str
from tqdm import tqdm
from clipS import clip
from clipS.model import *

def logits4pred(criterion, logits_list):
    if criterion.__class__.__name__.lower() in ['bceloss']:
        logits = logits_list[0]
        probs = logits.sigmoid()
    else:
        assert False, f"{criterion.__class__.__name__.lower()} not exits"

    return probs, logits

#text_feats = np.load("/home/compu/doanhbc/upar_challenge/SOLIDER-PersonAttributeRecognition/feat_list_attributes.npy")
#text_feats = torch.from_numpy(text_feats).to('cuda')

def batch_trainer(cfg, args, epoch, model,ViT_model, model_ema, train_loader, criterion, criterion_c, optimizer, loss_w=[1, ], scheduler=None, tb_writer=None,des_non=None):
    model.train()
    ViT_model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']
    
    num_classes = len(des_non)
    batch= cfg.TRAIN.BATCH_SIZE
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    with torch.no_grad():
        word_embed=[]
        for i in range(i_ter):
            if i+1 != i_ter:
                des_batch = des_non[i*batch:(i+1)*batch]
            else:
                des_batch = des_non[i*batch:num_classes]
            destoken = clip.tokenize(des_batch,truncate=True).cuda()
            word_embed.append(ViT_model.encode_text(destoken).cuda().float())
        word_embed=torch.cat(word_embed,dim=0).cuda()

    
    for step, (imgs, gt_label, imgname,des_label) in enumerate(train_loader):
        iter_num = epoch * len(train_loader) + step

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits = model(imgs,ViT_model,word_embed)
        # train_logits, _ = model(imgs, gt_label)

        # loss_list, loss_mtr = criterion(train_logits, gt_label)
        train_loss_g = criterion(train_logits[0], gt_label)
        train_loss_p= criterion(train_logits[1], gt_label)#tr_logits_part
        train_loss_t= criterion(train_logits[2], gt_label)
        train_loss_s= criterion_c(train_logits[3],des_label)#train_logits[3]#
        train_loss=train_loss_g + train_loss_p+ train_loss_t + train_loss_s

        train_loss.backward()
    
        optimizer.step()
        
        optimizer.zero_grad()
        
        loss_meter.update(to_scalar(train_loss))
        
        gt_list.append(gt_label.cpu().numpy())
        # train_probs = torch.sigmoid(train_logits[0])#(+train_logits[1]+train_logits[2])/3
        #########################
        logits=[]
        for i,_ in enumerate(train_logits[:3]):
            logits.append(torch.sigmoid(train_logits[i]))
        logits=torch.stack(logits,dim=1)
        
        train_probs=torch.max(logits,dim=1)[0]
        
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 20
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',
            f'loss_g:{train_loss_g:.4f}',f'loss_p:{train_loss_p:.4f}',f'loss_t:{train_loss_t:.4f}',f'loss_s:{train_loss_s:.4f}',f'train_loss:{loss_meter.val:.4f}')#,
    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


def valid_trainer(cfg, args, epoch, model,ViT_model, valid_loader, criterion, loss_w=[1, ]):
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname, des_label) in enumerate(valid_loader):#,description
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            gt_label[gt_label == 2] = 0
            # label_v = label_v[0].cuda()
            valid_logits = model(imgs,ViT_model)#,group_arr
            valid_loss_g = criterion(valid_logits[0], gt_label)
           
            valid_loss_p = criterion(valid_logits[1], gt_label)
            valid_loss_t= criterion(valid_logits[2], gt_label)
            
            valid_loss=valid_loss_g + valid_loss_p  + valid_loss_t #valid_loss_agg#+ 0.5*valid_logits[3] #valid_loss_similary#+ 0.5*valid_loss_similary_p #+ #regularizer_loss #+loss_itc + loss_itc_part #
            # valid_probs = torch.sigmoid(valid_logits[0]) #+ valid_logits[1] +valid_logits[2])/3)
            
            ######################
            for i,_ in enumerate(valid_logits):
                valid_logits[i] = torch.sigmoid(valid_logits[i])
            logits=torch.stack(valid_logits,dim=1)
            # valid_probs= torch.mean(logits,dim=1)
            valid_probs= torch.max(logits,dim=1)[0] #torch.max(logits,dim=1)[0]
            
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    valid_loss_g,valid_loss_p,valid_loss_patch = 0.,0.,0.
    return valid_loss, gt_label, preds_probs#,valid_loss_g,valid_loss_p,valid_loss_patch
