import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from scipy.stats import wasserstein_distance
from scipy.linalg import hadamard
from loguru import logger
import torch.nn.functional as F
from model.loss import *
import itertools
from model.model_loader import *
from evaluate import mean_average_precision
from torch.nn import Parameter
from torch.autograd import Variable
from utils import *
import random
from PIL import ImageFilter
torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train(train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          batch_size,
          source_rate,
          ):

    model = load_model(arch, code_length,num_class,num_class)
    # logger.info(model)
    model.to(device)
    #model = nn.DataParallel(model,device_ids=[0,1])
    # if isinstance(model,torch.nn.DataParallel):
    #     model = model.module

    parameter_list = model.get_parameters() 
    optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)
    supcon = SupConLoss()
    # model = nn.DataParallel(model,device_ids=[0,1])

    
    model.train()

    k = 40
    high_map = 0
    warm_up = 20
    with torch.autograd.set_detect_anomaly(True):
         
        code_s_centroid = torch.randn((num_class,code_length))
        code_t_centroid = torch.randn((num_class,code_length))
        num_s_centroid = torch.zeros((num_class,1))
        num_t_centroid = torch.zeros((num_class,1))
        for epoch in range(max_iter):
            
            if epoch < warm_up:
                for batch_idx, ((data_s, _, target_s, index), (data_t, data_t_aug,target_t, index_t)) in\
                enumerate(zip(train_s_dataloader, train_t_dataloader)):
                    start = time.time()
                    data_s = data_s.to(device)
                    target_s = target_s.to(device)
                    optimizer.zero_grad()
                    logit_s, z_s, code_s = model(data_s)
                    
                    ## closed-set classifier
                    loss = nn.CrossEntropyLoss()(logit_s, target_s.argmax(1))
                    
                    loss.backward()
                    optimizer.step()
                    end = time.time()
                    optimizer.zero_grad()
            else:
                loss_contrast_ = 0
                loss_swav_ = 0
                contrast_iter = 0
                swav_iter = 0
                
                d_centroid = torch.zeros((num_class,1))
                
                for batch_idx, ((data_s, _, target_s, index), (data_t, data_t_aug,target_t, index_t)) in\
                    enumerate(zip(train_s_dataloader, train_t_dataloader)):
                    start = time.time()
                    data_s = data_s.to(device)
                    target_s = target_s.to(device)
                    data_t = data_t.to(device)
                    #data_t_aug = data_t_aug.to(device)
                    optimizer.zero_grad()
                    logit_s, z_s, code_s = model(data_s)
                    logit_t, z_t, code_t = model(data_t)  #z_t [bat,4096]
                    #logit_t_aug, z_t_aug, code_t_aug = model(data_t_aug)
                    # -------------------------(1)Source's OrH loss---------------------------
                    ## closed-set classifier
                    loss_s = nn.CrossEntropyLoss()(logit_s, target_s.argmax(1))
                    # -------------------------(1)Source's OrH loss---------------------------
                    # -------------------------(2)Pairwise Similarity-------------------------
                    
                    if z_s.shape[0] == z_t.shape[0]:
                        topk = 4
                        prob_t_aug, prob_t = F.softmax(logit_t.detach().cpu(), dim=1), F.softmax(logit_s.detach().cpu(), dim=1)
                        rank_t_aug = prob_t_aug
                        rank_idx_t_aug = torch.argsort(rank_t_aug, dim=1, descending=True)
                        rank_idxt_aug = rank_idx_t_aug.repeat(rank_idx_t_aug.size(0),1)
                        rank_t = prob_t
                        rank_idx_t = torch.argsort(rank_t, dim=1, descending=True)
                        rank_idxt = rank_idx_t.repeat(1,rank_idx_t.size(0)).view(-1,rank_idx_t.size(1))
                        rank_idx1, rank_idx2 = rank_idxt_aug[:, :topk], rank_idxt[:, :topk]
                        rank_idx1, _ = torch.sort(rank_idx1)
                        rank_idx2, _ = torch.sort(rank_idx2)
                        rank_diff = rank_idx1 - rank_idx2
                        rank_diff = torch.mean(torch.abs(rank_diff).float(), dim=1)
                        target_ulb = torch.ones_like(rank_diff).float().to(device)
                        target_ulb[rank_diff > 0] = 0
                        S = target_ulb.view(logit_t.shape[0],-1)

                        threshold = 0.95
                        logit_t_ = F.normalize(logit_t.detach().cpu(), dim=-1)
                        logit_s_ = F.normalize(logit_s.detach().cpu(), dim=-1)
                        attn_tmp = torch.mm(logit_t_, logit_s_.transpose(0, 1))
                        zero_vec = -9e15 * torch.ones_like(attn_tmp)
                        attn_tmp = torch.where(attn_tmp > threshold, attn_tmp, zero_vec)
                        attn_tmp = F.softmax(attn_tmp, dim=-1)
                        logit_t_ = torch.mm(attn_tmp, logit_t_).to(device)
                        logit_s_ = logit_s_.to(device)
                        loss_supcon = supcon(torch.cat([logit_t_.unsqueeze(1),logit_s_.unsqueeze(1)],dim=1), mask=S)
                        loss_contrast_ += loss_supcon
                        contrast_iter +=1                
                    else:
                        loss_supcon = 0
                    
                    # -------------------------(2)Pairwise Similarity-------------------------
                    
                    
                    # -------------------------(3)Novel     Partition-------------------------
                    loss_swav = 0
                    if logit_s.shape[0] == logit_t.shape[0]:
                        for k in range(code_s.shape[0]):
                            cl = target_s.argmax(1)[k]
                            code_s_centroid[cl,:] +=  code_s[k,:].detach().cpu()
                            num_s_centroid[cl] +=  1
                        minum = torch.ones_like(num_s_centroid)
                        num_s_centroid = torch.where(num_s_centroid > 0,num_s_centroid,minum)
                        code_s_centroid /= num_s_centroid.repeat(1,code_length)

                        for cl in range(num_class):
                            d_min = 10000000000
                            for num in range(code_t.shape[0]):
                                d = wasserstein_distance(code_s_centroid[cl,:].detach().clone(),code_t[num,:].detach().cpu().clone())
                                if d<=d_min:
                                    d_min = d
                            d_centroid[cl] = d_min
                        d_,idx = torch.sort(d_centroid,dim=0)
                        common_cls = idx[:int(source_rate*num_class)]
                        novel_cls = idx[int(source_rate*num_class):]

                        target_t = logit_t.argmax(dim=1)
                        for k in range(code_t.shape[0]):
                            y_t = target_t[k].detach().cpu()
                            if y_t in common_cls:
                                # code_t_centroid[y_t] = code_t_centroid[y_t,:] + code_s_centroid[y_t,:]
                                # num_t_centroid[y_t] = num_t_centroid[y_t] + 1
                                code_t_centroid[y_t] += code_t[k,:].detach().cpu() + code_s_centroid[y_t,:]
                                num_t_centroid[y_t] = num_t_centroid[y_t] + 2
                                
                            else:
                                code_t_centroid[y_t,:] = code_t_centroid[y_t,:] + code_t[k,:].detach().cpu()
                                num_t_centroid[y_t] = num_t_centroid[y_t] + 1
                        minum_t = torch.ones_like(num_t_centroid)
                        num_t_centroid = torch.where(num_t_centroid > 0,num_t_centroid,minum_t)
                        code_t_centroid = code_t_centroid/num_t_centroid.repeat(1,code_length)
                    
                        code_all = torch.concat([code_t,code_s])
                        output = F.normalize(code_all.detach().cpu().mm(code_t_centroid.T))
                        output = output.to(device)
                        # nmb_crops = [2]
                        
                        temperature = 0.1
                        crop = [0,1]
                        for i, crop_id in enumerate(crop):
                            with torch.no_grad():
                                out = output[batch_size * crop_id: batch_size * (crop_id + 1)]
                                # get assignments
                                q = distributed_sinkhorn(out)#[-batch_size:]
                        # cluster assignment prediction
                            subloss = 0
                            x = output[batch_size * crop_id: batch_size * (crop_id + 1)] / temperature
                            subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                            loss_swav += subloss 
                        loss_swav /= len(crop)
                        loss_swav_ += loss_swav
                        swav_iter += 1

                    total_loss = loss_s + 0.5*loss_supcon +0.05 * loss_swav
    
                    total_loss.backward(retain_graph=True)
                    optimizer.step()
                    end = time.time()
                    optimizer.zero_grad()
            # if contrast_iter>0:    
            #     logger.info('Supervised Contrastive Loss{:.8f}'.format(loss_contrast_/contrast_iter))
            if epoch >= warm_up:
                #logger.info('SwAV Loss: {:.8f}'.format(loss_swav_/swav_iter))
                logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, total_loss.item()))
            else:
                logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, loss.item()))
            
            # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                save = True,
                                )
                if high_map <= mAP:
                    high_map = mAP
                logger.info('[iter:{}/{}][map:{:.4f}][highest map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                    high_map
                ))
            
        

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=True,
                   )
    # torch.save({'iteration': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    
    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _,index in dataloader:
            data = data.to(device)
            _,_,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code

def print_image(data, name):
    from PIL import Image
    im = Image.fromarray(data)
    im.save('fig/topk/{}.png'.format(name))








    


    
