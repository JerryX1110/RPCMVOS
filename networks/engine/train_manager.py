import os
import importlib
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np
from dataloaders.datasets import DAVIS2017_Train, YOUTUBE_VOS_Train, TEST
import dataloaders.custom_transforms as tr
from networks.deeplab.deeplab import DeepLab
from utils.meters import AverageMeter
from utils.image import label2colormap, masked_image, save_image
from utils.checkpoint import load_network_and_optimizer, load_network, save_network
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.metric import pytorch_iou
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
class Trainer(object):
    def __init__(self , rank, cfg):
        self.gpu = rank + cfg.DIST_START_GPU
        self.rank = rank
        self.cfg = cfg
        self.print_log(cfg.__dict__)
        print("Use GPU {} for training".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        self.print_log('Build backbone.')
        self.feature_extracter = DeepLab(
            backbone=cfg.MODEL_BACKBONE,
            freeze_bn=cfg.MODEL_FREEZE_BN).cuda(self.gpu)

        if cfg.MODEL_FREEZE_BACKBONE:
            for param in self.feature_extracter.parameters():
                param.requires_grad = False

        self.print_log('Build VOS model.')
        CFBI = importlib.import_module(cfg.MODEL_MODULE)

        self.model = CFBI.get_module()(
            cfg,
            self.feature_extracter).cuda(self.gpu)

        if cfg.DIST_ENABLE:
            dist.init_process_group(
                backend=cfg.DIST_BACKEND, 
                init_method=cfg.DIST_URL,
                world_size=cfg.TRAIN_GPUS, 
                rank=rank, 
                timeout=datetime.timedelta(seconds=300))
            self.dist_model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.gpu],
                find_unused_parameters=True)
        else:
            self.dist_model = self.model

        self.print_log('Build optimizer.')
        trainable_params = get_trainable_params(
            model=self.dist_model, 
            base_lr=cfg.TRAIN_LR, 
            weight_decay=cfg.TRAIN_WEIGHT_DECAY, 
            beta_wd=cfg.MODEL_GCT_BETA_WD)

        self.optimizer = optim.SGD(
            trainable_params, 
            lr=cfg.TRAIN_LR, 
            momentum=cfg.TRAIN_MOMENTUM, 
            nesterov=True)

        self.prepare_dataset()
        self.process_pretrained_model()

        if cfg.TRAIN_TBLOG and self.rank == 0:
            from tensorboardX import SummaryWriter
            self.tblogger = SummaryWriter(cfg.DIR_TB_LOG)

    def process_pretrained_model(self):
        cfg = self.cfg

        self.step = cfg.TRAIN_START_STEP
        self.epoch = 0

        if cfg.TRAIN_AUTO_RESUME:
            ckpts = os.listdir(cfg.DIR_CKPT)
            if len(ckpts) > 0:
                ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                ckpt = np.sort(ckpts)[-1]
                cfg.TRAIN_RESUME = True
                cfg.TRAIN_RESUME_CKPT = ckpt
                cfg.TRAIN_RESUME_STEP = ckpt + 1
            else:
                cfg.TRAIN_RESUME = False

        if cfg.TRAIN_RESUME:
            resume_ckpt = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))

            self.model, self.optimizer, removed_dict = load_network_and_optimizer(self.model, self.optimizer, resume_ckpt, self.gpu)

            if len(removed_dict) > 0:
                self.print_log('Remove {} from checkpoint.'.format(removed_dict))

            self.step = cfg.TRAIN_RESUME_STEP
            if cfg.TRAIN_TOTAL_STEPS <= self.step:
                self.print_log("Your training has finished!")
                exit()
            self.epoch = int(np.ceil(self.step / len(self.trainloader)))

            self.print_log('Resume from step {}'.format(self.step))

        elif cfg.PRETRAIN:
            if cfg.PRETRAIN_FULL:
                self.model, removed_dict = load_network(self.model, cfg.PRETRAIN_MODEL, self.gpu)
                if len(removed_dict) > 0:
                    self.print_log('Remove {} from pretrained model.'.format(removed_dict))
                self.print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAIN_MODEL))
            else:
                feature_extracter, removed_dict = load_network(self.feature_extracter, cfg.PRETRAIN_MODEL, self.gpu)
                if len(removed_dict) > 0:
                    self.print_log('Remove {} from pretrained model.'.format(removed_dict))
                self.print_log('Load pretrained backbone model from {}.'.format(cfg.PRETRAIN_MODEL))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        composed_transforms = transforms.Compose([
            tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR, cfg.DATA_MAX_SCALE_FACTOR, cfg.DATA_SHORT_EDGE_LEN),
            tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP), 
            tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
            tr.Resize(cfg.DATA_RANDOMCROP),
            tr.ToTensor()])
        
        train_datasets = []
        if 'davis2017' in cfg.DATASETS:
            train_davis_dataset = DAVIS2017_Train(
                root=cfg.DIR_DAVIS, 
                full_resolution=cfg.TRAIN_DATASET_FULL_RESOLUTION,
                transform=composed_transforms, 
                repeat_time=cfg.DATA_DAVIS_REPEAT,
                curr_len=cfg.DATA_CURR_SEQ_LEN,
                rand_gap=cfg.DATA_RANDOM_GAP_DAVIS,
                rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ)
            train_datasets.append(train_davis_dataset)

        if 'youtubevos' in cfg.DATASETS:
            train_ytb_dataset = YOUTUBE_VOS_Train(
                root=cfg.DIR_YTB, 
                transform=composed_transforms,
                curr_len=cfg.DATA_CURR_SEQ_LEN,
                rand_gap=cfg.DATA_RANDOM_GAP_YTB,
                rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ)
            train_datasets.append(train_ytb_dataset)

        if 'test' in cfg.DATASETS:
            test_dataset = TEST(
                transform=composed_transforms,
                curr_len=cfg.DATA_CURR_SEQ_LEN)
            train_datasets.append(test_dataset)

        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=int(cfg.TRAIN_BATCH_SIZE / cfg.TRAIN_GPUS),
            shuffle=False,
            num_workers=cfg.DATA_WORKERS, 
            pin_memory=True, 
            sampler=self.train_sampler)

        self.print_log('Done!')

    def sequential_training(self):
        
        cfg = self.cfg

        running_losses = []
        running_ious = []
        for _ in range(cfg.DATA_CURR_SEQ_LEN):
            running_losses.append(AverageMeter())
            running_ious.append(AverageMeter())
        batch_time = AverageMeter()
        avg_obj =  AverageMeter()       

        optimizer = self.optimizer
        model = self.dist_model
        train_sampler = self.train_sampler
        trainloader = self.trainloader
        step = self.step
        epoch = self.epoch
        max_itr = cfg.TRAIN_TOTAL_STEPS

        PlaceHolder=[]
        for i in range(cfg.BLOCK_NUM):
            PlaceHolder.append(None)

        self.print_log('Start training.')
        model.train()
        while step < cfg.TRAIN_TOTAL_STEPS:
            train_sampler.set_epoch(epoch)
            epoch += 1
            last_time = time.time()
            for frame_idx, sample in enumerate(trainloader):
                now_lr = adjust_learning_rate(
                    optimizer=optimizer, 
                    base_lr=cfg.TRAIN_LR, 
                    p=cfg.TRAIN_POWER, 
                    itr=step, 
                    max_itr=max_itr, 
                    warm_up_steps=cfg.TRAIN_WARM_UP_STEPS, 
                    is_cosine_decay=cfg.TRAIN_COSINE_DECAY)

                ref_imgs = sample['ref_img']  # batch_size * 3 * h * w
                prev_imgs = sample['prev_img']
                curr_imgs = sample['curr_img'][0]
                ref_labels = sample['ref_label']  # batch_size * 1 * h * w
                prev_labels = sample['prev_label']
                curr_labels = sample['curr_label'][0]
                obj_nums = sample['meta']['obj_num']
                bs, _, h, w = curr_imgs.size()

                ref_labels = ref_labels.cuda(self.gpu)
                prev_labels = prev_labels.cuda(self.gpu)
                curr_labels = curr_labels.cuda(self.gpu)
                obj_nums = obj_nums.cuda(self.gpu)
                 
                if step % cfg.TRAIN_TBLOG_STEP == 0 and self.rank == 0 and cfg.TRAIN_TBLOG:
                    tf_board = True
                else:
                    tf_board = False

                # Sequential training
                all_boards = []
                curr_imgs = prev_imgs
                curr_labels = prev_labels
                all_pred = prev_labels.squeeze(1)
                optimizer.zero_grad()
                memory_cur_list=[]
                memory_prev_list=[]
                for iii in range(int(cfg.TRAIN_BATCH_SIZE//cfg.TRAIN_GPUS)):
                    memory_cur_list.append(PlaceHolder)
                    memory_prev_list.append(PlaceHolder)

                for idx in range(cfg.DATA_CURR_SEQ_LEN):
                    prev_imgs = curr_imgs
                    curr_imgs = sample['curr_img'][idx]
                    inputs = torch.cat((ref_imgs, prev_imgs, curr_imgs), 0).cuda(self.gpu)
                    if step > cfg.TRAIN_START_SEQ_TRAINING_STEPS:
                        # Use previous prediction instead of ground-truth mask
                        prev_labels = all_pred.unsqueeze(1)
                    else:
                        # Use previous ground-truth mask
                        prev_labels = curr_labels
                    curr_labels = sample['curr_label'][idx].cuda(self.gpu)
                    
                    loss, all_pred, boards,memory_cur_list = model(
                        inputs,
                        memory_prev_list,
                        ref_labels,
                        prev_labels,
                        curr_labels,
                        gt_ids=obj_nums,
                        step=step,
                        tf_board=tf_board)

                    memory_prev_list = memory_cur_list

                    iou = pytorch_iou(all_pred.unsqueeze(1), curr_labels, obj_nums)
                    loss = torch.mean(loss) / cfg.DATA_CURR_SEQ_LEN
                    loss.backward()
                    all_boards.append(boards)
                    running_losses[idx].update(loss.item() * cfg.DATA_CURR_SEQ_LEN)
                    running_ious[idx].update(iou.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN_CLIP_GRAD_NORM)
                optimizer.step()
                batch_time.update(time.time() - last_time)
                avg_obj.update(obj_nums.float().mean().item())
                last_time = time.time()

                if step % cfg.TRAIN_TBLOG_STEP == 0 and self.rank == 0:
                    self.process_log(
                        ref_imgs, prev_imgs, curr_imgs, 
                        ref_labels, prev_labels, curr_labels, 
                        all_pred, all_boards, running_losses, running_ious, now_lr, step)

                if step % cfg.TRAIN_LOG_STEP == 0 and self.rank == 0:
                    strs = 'Itr:{}, LR:{:.7f}, Time:{:.3f}, Obj:{:.1f}'.format(step, now_lr, batch_time.avg, avg_obj.avg)
                    batch_time.reset()
                    avg_obj.reset()
                    for idx in range(cfg.DATA_CURR_SEQ_LEN):
                        strs += ', S{}: L {:.3f}({:.3f}) IoU {:.3f}({:.3f})'.format(idx, running_losses[idx].val, running_losses[idx].avg, 
                                                                                       running_ious[idx].val, running_ious[idx].avg)
                        running_losses[idx].reset()
                        running_ious[idx].reset()

                    self.print_log(strs)

                if step % cfg.TRAIN_SAVE_STEP == 0 and step != 0 and self.rank == 0:
                    self.print_log('Save CKPT (Step {}).'.format(step))
                    save_network(self.model, optimizer, step, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)

                step += 1
                if step > cfg.TRAIN_TOTAL_STEPS:
                    break
                
        if self.rank == 0:
            self.print_log('Save final CKPT (Step {}).'.format(step - 1))
            save_network(self.model, optimizer, step - 1, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)

    def print_log(self, string):
        if self.rank == 0:
            print(string)


    def process_log(self, 
            ref_imgs, prev_imgs, curr_imgs, 
            ref_labels, prev_labels, curr_labels, 
            curr_pred, all_boards, running_losses, running_ious, now_lr, step):
        cfg = self.cfg

        mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
        sigma = np.array([[[0.229]], [[0.224]], [[0.225]]])

        show_ref_img, show_prev_img, show_curr_img = [img.cpu().numpy()[0] * sigma + mean for img in [ref_imgs, prev_imgs, curr_imgs]]

        show_gt, show_prev_gt, show_ref_gt, show_preds_s = [label.cpu()[0].squeeze(0).numpy() for label in [curr_labels, prev_labels, ref_labels, curr_pred]]

        show_gtf, show_prev_gtf, show_ref_gtf, show_preds_sf = [label2colormap(label).transpose((2,0,1)) for label in [show_gt, show_prev_gt, show_ref_gt, show_preds_s]]

        if cfg.TRAIN_IMG_LOG or cfg.TRAIN_TBLOG:

            show_ref_img = masked_image(show_ref_img, show_ref_gtf, show_ref_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(show_ref_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_ref_img.jpeg' % (step)))

            show_prev_img = masked_image(show_prev_img, show_prev_gtf, show_prev_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(show_prev_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_prev_img.jpeg' % (step)))

            show_img_pred = masked_image(show_curr_img, show_preds_sf, show_preds_s)
            if cfg.TRAIN_IMG_LOG:
                save_image(show_img_pred, os.path.join(cfg.DIR_IMG_LOG, '%06d_prediction.jpeg' % (step)))

            show_curr_img = masked_image(show_curr_img, show_gtf, show_gt)
            if cfg.TRAIN_IMG_LOG:
                save_image(show_curr_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_groundtruth.jpeg' % (step)))

            if cfg.TRAIN_TBLOG:
                for seq_step, running_loss, running_iou in zip(range(len(running_losses)), running_losses, running_ious):
                    self.tblogger.add_scalar('S{}/Loss'.format(seq_step), running_loss.avg, step)
                    self.tblogger.add_scalar('S{}/IoU'.format(seq_step), running_iou.avg, step)

                self.tblogger.add_scalar('LR', now_lr, step)
                self.tblogger.add_image('Ref/Image', show_ref_img, step)
                self.tblogger.add_image('Ref/GT', show_ref_gtf, step)

                self.tblogger.add_image('Prev/Image', show_prev_img, step)
                self.tblogger.add_image('Prev/GT', show_prev_gtf, step)

                self.tblogger.add_image('Curr/Image_GT', show_curr_img, step)
                self.tblogger.add_image('Curr/Image_Pred', show_img_pred, step)

                self.tblogger.add_image('Curr/Mask_GT', show_gtf, step)
                self.tblogger.add_image('Curr/Mask_Pred', show_preds_sf, step)

                for seq_step, boards in enumerate(all_boards):
                    for key in boards['image'].keys():
                        tmp = boards['image'][key].cpu().numpy()
                        self.tblogger.add_image('S{}/' + key, tmp, step)
                    for key in boards['scalar'].keys():
                        tmp = boards['scalar'][key].cpu().numpy()
                        self.tblogger.add_scalar('S{}/' + key, tmp, step)

                self.tblogger.flush()

        del(all_boards)


