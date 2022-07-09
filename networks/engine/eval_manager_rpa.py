import os
import importlib
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np
from dataloaders.datasets import YOUTUBE_VOS_Test,DAVIS_Test
import dataloaders.custom_transforms as tr
from networks.deeplab.deeplab import DeepLab
from utils.meters import AverageMeter
from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder
from networks.layers.shannon_entropy import cal_shannon_entropy
import math

class Evaluator(object):
    def __init__(self, cfg):

        self.mem_every = cfg.MEM_EVERY
        self.unc_ratio = cfg.UNC_RATIO
        
        self.gpu = cfg.TEST_GPU_ID
        self.cfg = cfg
        self.print_log(cfg.__dict__)
        print("Use GPU {} for evaluating".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        self.print_log('Build backbone.')
        self.feature_extracter = DeepLab(
            backbone=cfg.MODEL_BACKBONE,
            freeze_bn=cfg.MODEL_FREEZE_BN).cuda(self.gpu)

        self.print_log('Build VOS model.')
        RPCM = importlib.import_module(cfg.MODEL_MODULE)
        self.model = RPCM.get_module()(
            cfg,
            self.feature_extracter).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg
        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return
        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE, cfg.TEST_FLIP, cfg.TEST_MULTISCALE), 
            tr.MultiToTensor()])
        
        eval_name = '{}_{}_ckpt_{}'.format(cfg.TEST_DATASET, cfg.EXP_NAME, self.ckpt)
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms'
            for scale in cfg.TEST_MULTISCALE:
                eval_name +="_"
                eval_name +=str(scale)

        eval_name += "_mem_"+str(self.mem_every)+"_unc_"+str(self.unc_ratio)+"_res_"+str(cfg.TEST_MAX_SIZE)+"_wRPA"
        
        if cfg.TEST_DATASET == 'youtubevos19':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = YOUTUBE_VOS_Test(
                root=cfg.DIR_YTB_EVAL19, 
                transform=eval_transforms,  
                result_root=self.result_root)
            
        elif cfg.TEST_DATASET == 'youtubevos18':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = YOUTUBE_VOS_Test(
                root=cfg.DIR_YTB_EVAL18, 
                transform=eval_transforms,  
                result_root=self.result_root)
        else:
            print('Unknown dataset!')
            exit()

        print('Eval {} on {}:'.format(cfg.EXP_NAME, cfg.TEST_DATASET))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0 
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        PlaceHolder=[]
        for i in range(cfg.BLOCK_NUM):
            PlaceHolder.append(None)

        for seq_idx, seq_dataset in enumerate(self.dataset):
            video_num += 1
            seq_name = seq_dataset.seq_name

            print('Prcessing Seq {} [{}/{}]:'.format(seq_name, video_num, total_video_num))

            torch.cuda.empty_cache()

            seq_dataloader=DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=cfg.TEST_WORKERS, pin_memory=True)
            
            seq_total_time = 0
            seq_total_frame = 0
            ref_embeddings = []
            ref_masks = []
            prev_embedding = []
            prev_mask = []
            ref_mask_confident = [] 
            memory_prev_all_list=[]
            memory_cur_all_list=[]
            memory_prev_list=[]
            memory_cur_list=[]
            label_all_list=[]
            
            with torch.no_grad():
                for frame_idx, samples in enumerate(seq_dataloader):

                    time_start = time.time()
                    all_preds = []

                    join_label = None
                    UPDATE=False


                    if frame_idx==0:
                        for aug_idx in range(len(samples)):
                            memory_prev_all_list.append([PlaceHolder])
                    else:
                        memory_prev_all_list=memory_cur_all_list

                    memory_cur_all_list=[]
                    for aug_idx in range(len(samples)):
                        if len(ref_embeddings) <= aug_idx:
                            ref_embeddings.append([])
                            ref_masks.append([])
                            prev_embedding.append(None)
                            prev_mask.append(None)
                            ref_mask_confident.append([])

                        sample = samples[aug_idx]
                        ref_emb = ref_embeddings[aug_idx]

                        ## use confident mask for correlation
                        ref_m = ref_mask_confident[aug_idx]  

                        prev_emb = prev_embedding[aug_idx]
                        prev_m = prev_mask[aug_idx]
                        

                        current_img = sample['current_img']
                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(self.gpu)
                        else:
                            current_label = None

                        obj_list = sample['meta']['obj_list']
                        obj_num = sample['meta']['obj_num']
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        current_img = current_img.cuda(self.gpu)
                        obj_num = obj_num.cuda(self.gpu)
                        bs, _, h, w = current_img.size()

                        all_pred, current_embedding,memory_cur_list = self.model.forward_for_eval(memory_prev_all_list[aug_idx], ref_emb, 
                                                                                                  ref_m, prev_emb, prev_m, 
                                                                                                  current_img, gt_ids=obj_num, 
                                                                                                  pred_size=[ori_height,ori_width])
                        memory_cur_all_list.append(memory_cur_list)

                        # delete the label that hasn't existed in the GT label for YTB-VOS
                        all_pred_remake = []
                        all_pred_exist = []
                        if all_pred!=None:
                            all_pred_split = all_pred.split(all_pred.size()[1],dim=1)[0]

                            for i in range(all_pred.size()[1]):
                                if i not in label_all_list:
                                    all_pred_remake.append(torch.zeros_like(all_pred_split[0][i]).unsqueeze(0))
                                else:
                                    all_pred_remake.append(all_pred_split[0][i].unsqueeze(0))
                                    all_pred_exist.append(all_pred_split[0][i].unsqueeze(0))
                            all_pred = torch.cat(all_pred_remake,dim=0).unsqueeze(0)
                            all_pred_exist = torch.cat(all_pred_exist,dim=0).unsqueeze(0)


                        if 'current_label' in sample.keys():
                            label_cur_list  = np.unique(sample['current_label'].cpu().detach().numpy()).tolist()
                            for i in label_cur_list:
                                if i not in label_all_list:
                                    label_all_list.append(i)

                        if frame_idx == 0:
                            if current_label is None:
                                print("No first frame label in Seq {}.".format(seq_name))
                            ref_embeddings[aug_idx].append(current_embedding)
                            ref_masks[aug_idx].append(current_label)
                            ref_mask_confident[aug_idx].append(current_label)

                            prev_embedding[aug_idx] = current_embedding
                            prev_mask[aug_idx] = current_label
                            
                        else:
                            if sample['meta']['flip']:
                                all_pred = flip_tensor(all_pred, 3)

                            #  In YouTube-VOS, not all the objects appear in the first frame for the first time. Thus, we
                            #  have to introduce new labels for new objects, if necessary.
                            if not sample['meta']['flip'] and not(current_label is None) and join_label is None: # gt exists here
                                join_label = current_label
                            all_preds.append(all_pred)
                            
                            all_pred_org = all_pred
                            current_label_0 = None

                            if current_label is not None:
                                ref_embeddings[aug_idx].append(current_embedding)
                                
                            else:
                                all_preds_0 = torch.cat(all_preds, dim=0)
                                all_preds_0 = torch.mean(all_preds_0, dim=0)
                                pred_label_0 = torch.argmax(all_preds_0, dim=0)
                                current_label_0 = pred_label_0.view(1, 1, ori_height, ori_width)

                                # uncertainty region filter
                                uncertainty_org,uncertainty_norm = cal_shannon_entropy(all_pred_exist)       

                                # we set mem_every == -1 to indicate we don't use extra confident candidate pool
                                if self.mem_every>-1 and frame_idx % self.mem_every==0 and frame_idx!=0 and current_embedding!=None and current_label_0!=None:
                                    ref_embeddings[aug_idx].append(current_embedding)
                                    ref_masks[aug_idx].append(current_label_0)
                                    UPDATE=True


                            prev_embedding[aug_idx] = current_embedding

                    if frame_idx > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        all_preds = torch.mean(all_preds, dim=0)
                        pred_label = torch.argmax(all_preds, dim=0)
                        if join_label is not None:
                            join_label = join_label.squeeze(0).squeeze(0)
                            keep = (join_label == 0).long()
                            pred_label = pred_label * keep + join_label * (1 - keep)
                            pred_label = pred_label
                        current_label = pred_label.view(1, 1, ori_height, ori_width)
                        if samples[aug_idx]['meta']['flip']:
                            flip_pred_label = flip_tensor(pred_label, 1)
                            flip_current_label = flip_pred_label.view(1, 1, ori_height, ori_width)

                        for aug_idx in range(len(samples)):
                            if join_label is not None:
                                if samples[aug_idx]['meta']['flip']:
                                    ref_masks[aug_idx].append(flip_current_label)
                                    ref_mask_confident[aug_idx].append(flip_current_label)
                                else:
                                    ref_masks[aug_idx].append(current_label)

                                    uncertainty_org,uncertainty_norm = cal_shannon_entropy(all_pred_exist) 
                                    join_label = join_label.squeeze(0).squeeze(0)
                                    keep = (join_label == 0).long() 
                                    join_uncertainty_map =  (join_label <0).long()
                                    uncertainty_org = uncertainty_org * keep + join_uncertainty_map * (1 - keep)

                                    uncertainty_region = (uncertainty_org>self.unc_ratio ).long()

                                    # we use 125 to represent the filtered patches
                                    pred_label_c = pred_label*  (1 - uncertainty_region)  + (125) * uncertainty_region
                                    pred_label_c = pred_label_c.view(1, 1, ori_height, ori_width)

                                    ref_mask_confident[aug_idx].append(pred_label_c)

                            if samples[aug_idx]['meta']['flip']:
                                prev_mask[aug_idx] = flip_current_label
                            else:
                                prev_mask[aug_idx] = current_label
                                
                            if UPDATE:
                                if self.mem_every>-1 and frame_idx%self.mem_every==0 and frame_idx!=0 and current_embedding!=None and current_label_0!=None :
                                    uncertainty_region = (uncertainty_org>self.unc_ratio ).long()
                                    pred_label_c = pred_label*  (1 - uncertainty_region)  + (125) * uncertainty_region
                                    pred_label_c = pred_label_c.view(1, 1, ori_height, ori_width)
                                    ref_mask_confident[aug_idx].append(pred_label_c)

                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        seq_total_frame += 1
                        obj_num = obj_num[0].item()
                        print('Frame: {}, Obj Num: {}, Time: {}'.format(imgname[0], obj_num, one_frametime))
                        # Save result
                        save_mask(pred_label, os.path.join(self.result_root, seq_name, imgname[0].split('.')[0]+'.png'))

                    else:
                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        print('Ref Frame: {}, Time: {}'.format(imgname[0], one_frametime))

                del(ref_embeddings)
                del(ref_masks)
                del(prev_embedding)
                del(prev_mask)
                del(seq_dataset)
                del(seq_dataloader)
                del(memory_cur_all_list)


            seq_avg_time_per_frame = seq_total_time / seq_total_frame
            total_time += seq_total_time
            total_frame += seq_total_frame
            total_avg_time_per_frame = total_time / total_frame
            total_sfps += seq_avg_time_per_frame
            avg_sfps = total_sfps / (seq_idx + 1)
            print("Seq {} FPS: {}, Total FPS: {}, FPS per Seq: {}".format(seq_name, 1./seq_avg_time_per_frame, 1./total_avg_time_per_frame, 1./avg_sfps))

        zip_folder(self.source_folder, self.zip_dir)
        self.print_log('Save result to {}.'.format(self.zip_dir))
        

    def print_log(self, string):
        print(string)





