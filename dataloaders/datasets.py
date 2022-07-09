from __future__ import division
import json
import os
import shutil
import numpy as np
import torch, cv2
from random import choice
from torch.utils.data import Dataset
import json
from PIL import Image
import random
from utils.image import _palette

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
        
    return Ms

class VOS_Train(Dataset):
    def __init__(self, 
            image_root,
            label_root,
            imglistdic,
            transform=None,
            rgb=False,
            repeat_time=1,
            rand_gap=3,
            curr_len=3,
            rand_reverse=True
            ):
        self.image_root = image_root
        self.label_root = label_root
        self.rand_gap = rand_gap
        self.curr_len = curr_len
        self.rand_reverse = rand_reverse
        self.repeat_time = repeat_time
        self.transform = transform
        self.rgb = rgb
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())
        print('Video num: {}'.format(len(self.seqs)))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def get_ref_index(self, seqname, lablist, objs, min_fg_pixels=200, max_try=5):
        for _ in range(max_try):
            ref_index = np.random.randint(len(lablist))
            ref_label = Image.open(os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            ref_objs = list(np.unique(ref_label))
            is_consistent = True
            for obj in ref_objs:
                if obj == 0:
                    continue
                if obj not in objs:
                    is_consistent = False
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and is_consistent:
                break
        return ref_index

    def get_ref_index_v2(self, seqname, lablist, min_fg_pixels=200, max_try=5):
        for _ in range(max_try):
            ref_index = np.random.randint(len(lablist))
            ref_label = Image.open(os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels:
                break
        return ref_index

    def get_curr_gaps(self):
        curr_gaps = []
        total_gap = 0
        for _ in range(self.curr_len):
            gap = int(np.random.randint(self.rand_gap) + 1)
            total_gap += gap
            curr_gaps.append(gap)
        return curr_gaps, total_gap

    def get_prev_index(self, lablist, total_gap):
        search_range = len(lablist) - total_gap
        if search_range > 1:
            prev_index = np.random.randint(search_range)
        else:
            prev_index = 0
        return prev_index

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def get_curr_indices(self, lablist, prev_index, gaps):
        total_len = len(lablist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index):
        image = cv2.imread(os.path.join(self.image_root, seqname, imagelist[index]))
        image = np.array(image, dtype=np.float32)
        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label = Image.open(os.path.join(self.label_root, seqname, lablist[index]))
        label = np.array(label, dtype=np.uint8)

        return image, label

    def __getitem__(self, idx):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while(is_consistent == False and try_step < max_try):
            try_step += 1
            # get prev frame
            curr_gaps, total_gap = self.get_curr_gaps()
            prev_index = self.get_prev_index(lablist, total_gap)
            prev_image, prev_label = self.get_image_label(seqname, imagelist, lablist, prev_index)
            prev_objs = list(np.unique(prev_label))

            # get curr frames
            curr_indices = self.get_curr_indices(lablist, prev_index, curr_gaps)
            curr_images, curr_labels, curr_objs = [], [], []
            for curr_index in curr_indices:
                curr_image, curr_label = self.get_image_label(seqname, imagelist, lablist, curr_index)
                c_objs = list(np.unique(curr_label))
                curr_images.append(curr_image)
                curr_labels.append(curr_label)
                curr_objs.extend(c_objs)

            objs = list(np.unique(prev_objs + curr_objs))
            # get ref frame
            ref_index = self.get_ref_index_v2(seqname, lablist)
            ref_image, ref_label = self.get_image_label(seqname, imagelist, lablist, ref_index)
            ref_objs = list(np.unique(ref_label))

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_num = list(np.sort(ref_objs))[-1]

        sample = {'ref_img':ref_image, 'prev_img':prev_image, 'curr_img':curr_images, 
                  'ref_label':ref_label,'prev_label':prev_label,'curr_label':curr_labels}
        sample['meta'] = {'seq_name':seqname, 'frame_num':frame_num, 'obj_num':obj_num}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class DAVIS2017_Train(VOS_Train):
    def __init__(self, 
            split=['train'],
            root='./DAVIS',
            transform=None,
            rgb=False,
            repeat_time=1,
            full_resolution=True,
            year=2017,
            rand_gap=3,
            curr_len=3,
            rand_reverse=True
            ):
        if full_resolution:
            resolution = 'Full-Resolution'
            if not os.path.exists(os.path.join(root, 'JPEGImages', resolution)):
                print('No Full-Resolution, use 480p instead.')
                resolution = '480p'
        else:
            resolution = '480p'
        image_root = os.path.join(root, 'JPEGImages', resolution)
        label_root = os.path.join(root, 'Annotations', resolution)
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets', str(year), spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        imglistdic = {}
        for seq_name in seq_names:
            images = list(np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(np.sort(os.listdir(os.path.join(label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        super(DAVIS2017_Train, self).__init__(
                                        image_root,
                                        label_root,
                                        imglistdic,
                                        transform,
                                        rgb,
                                        repeat_time,
                                        rand_gap,
                                        curr_len,
                                        rand_reverse)

class YOUTUBE_VOS_Train(VOS_Train):
    def __init__(self,
            root='./train',
            transform=None,
            rgb=False,
            rand_gap=3,
            curr_len=3,
            rand_reverse=True
            ):

        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        self.seq_list_file = os.path.join(root, 'meta.json')
        self._check_preprocess()
        seq_names = list(self.ann_f.keys())  

        imglistdic={}   
        for seq_name in seq_names:
            data = self.ann_f[seq_name]['objects']
            obj_names = list(data.keys())
            images = []
            labels = []
            for obj_n in obj_names:
                if len(data[obj_n]["frames"]) < 2:
                    print("Short object: " + seq_name + '-' + obj_n)
                    continue
                images += list(map(lambda x: x + '.jpg', list(data[obj_n]["frames"])))
                labels += list(map(lambda x: x + '.png', list(data[obj_n]["frames"])))
            images = np.sort(np.unique(images))
            labels = np.sort(np.unique(labels))
            if len(images) < 2:
                print("Short video: " + seq_name)
                continue
            imglistdic[seq_name] = (images, labels)

        super(YOUTUBE_VOS_Train, self).__init__(
                                        image_root,
                                        label_root,
                                        imglistdic,
                                        transform,
                                        rgb,
                                        1,
                                        rand_gap,
                                        curr_len,
                                        rand_reverse)

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print('No such file: {}.'.format(self.seq_list_file))
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True


class TEST(Dataset):

    def __init__(self,
            curr_len=3,
            obj_num=3,
            transform=None,
            ):
        self.curr_len = curr_len
        self.obj_num = obj_num
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        img = np.zeros((800, 800, 3)).astype(np.float32)
        label = np.ones((800, 800)).astype(np.uint8)
        sample = {'ref_img':img, 'prev_img':img, 'curr_img':[img]*self.curr_len, 
            'ref_label':label, 'prev_label':label, 'curr_label':[label]*self.curr_len}
        sample['meta'] = {'seq_name':'test', 'frame_num':100, 'obj_num':self.obj_num}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class _EVAL_TEST(Dataset):
    def __init__(self, transform, seq_name):
        self.seq_name = seq_name
        self.num_frame = 10
        self.transform = transform

    def __len__(self):
        return self.num_frame

    def __getitem__(self, idx):
        current_frame_obj_num = 2
        height = 400
        width = 400
        img_name = 'test{}.jpg'.format(idx)
        current_img = np.zeros((height, width, 3)).astype(np.float32)
        if idx == 0:
            current_label = (current_frame_obj_num * np.ones((height, width))).astype(np.uint8)
            sample = {'current_img':current_img, 'current_label':current_label}
        else:
            sample = {'current_img':current_img}

        sample['meta'] = {'seq_name':self.seq_name, 'frame_num':self.num_frame, 'obj_num':current_frame_obj_num,
                          'current_name':img_name, 'height':height, 'width':width, 'flip':False}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class EVAL_TEST(object):
    def __init__(self, transform=None, result_root=None):
        self.transform = transform
        self.result_root = result_root

        self.seqs = ['test1', 'test2', 'test3']

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]

        if not os.path.exists(os.path.join(self.result_root, seq_name)):
            os.makedirs(os.path.join(self.result_root, seq_name))

        seq_dataset = _EVAL_TEST(self.transform, seq_name)
        return seq_dataset

class VOS_Test(Dataset):
    def __init__(self, image_root, label_root, seq_name, images, labels, rgb=False, transform=None, single_obj=False, resolution=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb
        self.single_obj = single_obj
        self.resolution = resolution

        self.obj_nums = []
        self.objs = []
        temp_obj_num = 0
        obj_list_temp=[0]
        objs = []
        masks = []
        info = {}
        info['gt_obj'] = {}
        for img_name in self.images:
            self.obj_nums.append(temp_obj_num)
            objs.append(obj_list_temp)
            
            current_label_name = img_name.split('.')[0] + '.png'
            ### BUG BUG BUG
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                if temp_obj_num < np.unique(current_label)[-1]:
                    temp_obj_num = np.unique(current_label)[-1]

                label_list  = np.unique(current_label).tolist()
                for i in label_list:
                    if i!=0:
                        if i not in obj_list_temp:
                            obj_list_temp.append(i)
                
                current_path = os.path.join(self.label_root, self.seq_name, current_label_name)
                masks.append(np.array(Image.open(current_path).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                masks.append(np.zeros_like(masks[0]))
        self.objs = objs
        masks = np.stack(masks, 0)
        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = all_to_onehot(masks, labels)
        self.masks = masks
        self.info = info
        
        #print("self.masks.shape",self.masks.shape)
        
        

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        if self.single_obj:
            label = (label > 0).astype(np.uint8)
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        current_label_name = img_name.split('.')[0] + '.png'
        obj_num = self.obj_nums[idx]
        obj_list = self.objs[idx]
        
        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name)
            sample = {'current_img':current_img, 'current_label':current_label}
        else:
            sample = {'current_img':current_img}
        
        sample['meta'] = {'seq_name':self.seq_name, 'frame_num':self.num_frame, 'obj_num':obj_num,'obj_list':obj_list,
                          'current_name':img_name, 'height':height, 'width':width, 'flip':False}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class VOS_Test_all(Dataset):
    def __init__(self, image_root, label_root, seq_name, images, labels, rgb=False, transform=None, single_obj=False, resolution=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb
        self.single_obj = single_obj
        self.resolution = resolution

        self.obj_nums = []
        self.objs = []
        temp_obj_num = 0
        obj_list_temp=[0]
        objs = []
        masks = []
        info = {}
        info['gt_obj'] = {}
        for img_name in self.images:
            self.obj_nums.append(temp_obj_num)
            objs.append(obj_list_temp)
            
            current_label_name = img_name.split('.')[0] + '.png'
            ### BUG BUG BUG
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                if temp_obj_num < np.unique(current_label)[-1]:
                    temp_obj_num = np.unique(current_label)[-1]

                label_list  = np.unique(current_label).tolist()
                for i in label_list:
                    if i!=0:
                        if i not in obj_list_temp:
                            obj_list_temp.append(i)
                
                current_path = os.path.join(self.label_root, self.seq_name, current_label_name)
                masks.append(np.array(Image.open(current_path).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                masks.append(np.zeros_like(masks[0]))
        self.objs = objs
        masks = np.stack(masks, 0)
        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = all_to_onehot(masks, labels)
        self.masks = masks
        self.info = info
        
        #print("self.masks.shape",self.masks.shape)
        
        

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        if self.single_obj:
            label = (label > 0).astype(np.uint8)
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        current_label_name = img_name.split('.')[0] + '.png'
        obj_num = self.obj_nums[idx]
        obj_list = self.objs[idx]
        #print("len(self.labels)",len(self.labels))
        current_label_all = self.read_label(current_label_name)
        if current_label_name in self.labels:
            #print("current_label_name",current_label_name)
            #print(idx)
            current_label = self.read_label(current_label_name)
            """
            print("1current_label.shape")
            print(current_label)
            print(" np.unique(current_label).astype(np.uint8)", np.unique(current_label).astype(np.uint8))
            current_label = self.masks[0][idx] #self.read_label(current_label_name)
            print("2current_label.shape")
            print(current_label)
            print(" np.unique(current_label).astype(np.uint8)", np.unique(current_label).astype(np.uint8))
            """
            sample = {'current_img':current_img, 'current_label':current_label, 'current_label_all':current_label_all}
        else:
            sample = {'current_img':current_img, 'current_label_all':current_label_all}
        
        sample['meta'] = {'seq_name':self.seq_name, 'frame_num':self.num_frame, 'obj_num':obj_num,'obj_list':obj_list,
                          'current_name':img_name, 'height':height, 'width':width, 'flip':False}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class YOUTUBE_VOS_Test(object):
    def __init__(self, root='./valid', transform=None, rgb=False, result_root=None,use_all=False):

        self.db_root_dir = root
        self.result_root = result_root
        self.rgb = rgb
        self.transform = transform
        
        #self.seq_list_file = os.path.join(self.db_root_dir, 'meta.json')
        if use_all:
            self.seq_list_file = os.path.join(self.db_root_dir, 'meta_all.json')
        else:
            self.seq_list_file = os.path.join(self.db_root_dir, 'meta.json')
        self._check_preprocess()
        self.seqs = list(self.ann_f.keys())
        self.image_root = os.path.join(root, 'JPEGImages')
        self.label_root = os.path.join(root, 'Annotations')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        data = self.ann_f[seq_name]['objects']
        obj_names = list(data.keys())
        images = []
        labels = []
        for obj_n in obj_names:
            images += map(lambda x: x + '.jpg', list(data[obj_n]["frames"]))
            labels.append(data[obj_n]["frames"][0] + '.png')
        images = np.sort(np.unique(images))
        labels = np.sort(np.unique(labels))

        if not os.path.isfile(os.path.join(self.result_root, seq_name, labels[0])):
            if not os.path.exists(os.path.join(self.result_root, seq_name)):
                os.makedirs(os.path.join(self.result_root, seq_name))
            shutil.copy(os.path.join(self.label_root, seq_name, labels[0]), os.path.join(self.result_root, seq_name, labels[0]))
        images = np.sort(np.unique(images))
        labels = np.sort(np.unique(labels))
        seq_dataset = VOS_Test(self.image_root, self.label_root, seq_name, images, labels, transform=self.transform, rgb=self.rgb)
        return seq_dataset

    def _check_preprocess(self):
        _seq_list_file = self.seq_list_file
        if not os.path.isfile(_seq_list_file):
            print(_seq_list_file)
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True


class DAVIS_Test(object):
    def __init__(self, split=['val'], root='./DAVIS', year=2017, transform=None, rgb=False, full_resolution=False, result_root=None):
        self.transform = transform
        self.rgb = rgb
        self.result_root = result_root
        if year == 2016:
            self.single_obj = True
        else:
            self.single_obj = False
        if full_resolution:
            resolution = 'Full-Resolution'
        else:
            resolution = '480p'
        self.image_root = os.path.join(root, 'JPEGImages', resolution)
        self.label_root = os.path.join(root, 'Annotations', resolution)
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets', str(year), spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        self.seqs = list(np.unique(seq_names))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        images = list(np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
        labels = [images[0].replace('jpg', 'png')]

        if not os.path.isfile(os.path.join(self.result_root, seq_name, labels[0])):
            if not os.path.exists(os.path.join(self.result_root, seq_name)):
                os.makedirs(os.path.join(self.result_root, seq_name))
            source_label_path = os.path.join(self.label_root, seq_name, labels[0])
            result_label_path = os.path.join(self.result_root, seq_name, labels[0])
            if self.single_obj:
                label = Image.open(source_label_path)
                label = np.array(label, dtype=np.uint8)
                label = (label > 0).astype(np.uint8)
                label = Image.fromarray(label).convert('P')
                label.putpalette(_palette)
                label.save(result_label_path)
            else:
                shutil.copy(source_label_path, result_label_path)


        seq_dataset = VOS_Test(self.image_root, self.label_root, seq_name, images, labels, 
                               transform=self.transform, rgb=self.rgb, single_obj=self.single_obj, resolution=480)
        return seq_dataset
