import os
import random
import cv2
import numpy as np
import torch

cv2.setNumThreads(0)

class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        prev_img = sample['prev_img']
        h, w = prev_img.shape[:2]
        if self.output_size == (h, w):
            return sample
        else:
            new_h, new_w = self.output_size

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if elem == 'prev_img' or elem == 'curr_img' or elem == 'ref_img':
                flagval = cv2.INTER_CUBIC
            else:
                flagval = cv2.INTER_NEAREST

            if elem == 'curr_img' or elem == 'curr_label':
                new_tmp = []
                all_tmp = tmp
                for tmp in all_tmp:
                    tmp = cv2.resize(tmp, dsize=(new_w, new_h),
                          interpolation=flagval)
                    new_tmp.append(tmp)
                tmp = new_tmp
            else:
                tmp = cv2.resize(tmp, dsize=(new_w, new_h),
                          interpolation=flagval)

            sample[elem] = tmp

        return sample

class BalancedRandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, max_step=5, max_obj_num=5, min_obj_pixel_num=100):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.max_step = max_step
        self.max_obj_num = max_obj_num
        self.min_obj_pixel_num = min_obj_pixel_num

    def __call__(self, sample):

        image = sample['prev_img']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w
        ref_label = sample["ref_label"]
        prev_label = sample["prev_label"]
        curr_label = sample["curr_label"]
        
        is_contain_obj = False
        step = 0
        while (not is_contain_obj) and (step < self.max_step):
            step += 1
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            after_crop = []
            contains = []
            for elem in ([ref_label, prev_label] + curr_label):
                tmp = elem[top: top + new_h, left:left + new_w]
                contains.append(np.unique(tmp))
                after_crop.append(tmp)
            
            
            all_obj = list(np.sort(contains[0]))

            if all_obj[-1] == 0:
                continue
                
            # remove background
            if all_obj[0] == 0:
                all_obj = all_obj[1:]
            # remove small obj
            new_all_obj = []
            for obj_id in all_obj:
                after_crop_pixels = np.sum(after_crop[0] == obj_id)
                if after_crop_pixels > self.min_obj_pixel_num:
                    new_all_obj.append(obj_id)

            if len(new_all_obj) == 0:
                is_contain_obj = False
            else:
                is_contain_obj = True

            if len(new_all_obj) > self.max_obj_num:
                random.shuffle(new_all_obj)
                new_all_obj = new_all_obj[:self.max_obj_num]

            all_obj = [0] + new_all_obj


        post_process = []
        for elem in after_crop:
            new_elem = elem * 0
            for idx in range(len(all_obj)):
                obj_id = all_obj[idx]
                if obj_id == 0:
                    continue
                mask = elem == obj_id
                
                new_elem += (mask * idx).astype(np.uint8)
            post_process.append(new_elem.astype(np.uint8))

        sample["ref_label"] = post_process[0]
        sample["prev_label"] = post_process[1]
        curr_len = len(sample["curr_img"])
        sample["curr_label"] = []
        for idx in range(curr_len):
            sample["curr_label"].append(post_process[idx + 2])

        for elem in sample.keys():
            if 'meta' in elem or 'label' in elem:
                continue
            if elem == 'curr_img':
                new_tmp = []
                for tmp_ in sample[elem]:
                    tmp_ = tmp_[top: top + new_h, left:left + new_w]
                    new_tmp.append(tmp_)
                sample[elem] = new_tmp
            else:
                tmp = sample[elem]
                tmp = tmp[top: top + new_h, left:left + new_w]
                sample[elem] = tmp

        obj_num = len(all_obj) - 1

        sample['meta']['obj_num'] = obj_num

        return sample

class RandomScale(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, min_scale=1., max_scale=1.3, short_edge=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.short_edge = short_edge

    def __call__(self, sample):
        # Fixed range of scales
        sc = np.random.uniform(self.min_scale, self.max_scale)
        # Align short edge
        if not (self.short_edge is None):
            image = sample['prev_img']
            h, w = image.shape[:2]
            if h > w:
                sc *= float(self.short_edge) / w
            else:
                sc *= float(self.short_edge) / h


        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if elem == 'prev_img' or elem == 'curr_img' or elem == 'ref_img':
                flagval = cv2.INTER_CUBIC
            else:
                flagval = cv2.INTER_NEAREST

            if elem == 'curr_img' or elem == 'curr_label':
                new_tmp = []
                for tmp_ in tmp:
                    tmp_ = cv2.resize(tmp_, None, fx=sc, fy=sc, interpolation=flagval)
                    new_tmp.append(tmp_)
                tmp = new_tmp
            else:
                tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample

class RestrictSize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, min_size=None, max_size=800*1.3):
        self.min_size = min_size
        self.max_size = max_size
        assert ((min_size is None)) or ((max_size is None))

    def __call__(self, sample):

        # Fixed range of scales
        sc = None
        image = sample['ref_img']
        h, w = image.shape[:2]
        # Align short edge
        if not (self.min_size is None):
            if h > w:
                short_edge = w
            else:
                short_edge = h
            if short_edge < self.min_size:
                sc = float(self.min_size) / short_edge
        else:
            if h > w:
                long_edge = h
            else:
                long_edge = w
            if long_edge > self.max_size:
                sc = float(self.max_size) / long_edge

        if sc is None:
            new_h = h
            new_w = w
        else:
            new_h = int(sc * h)
            new_w = int(sc * w)
        new_h = new_h - (new_h - 1) % 4
        new_w = new_w - (new_w - 1) % 4
        if new_h == h and new_w == w:
            return sample

            
        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if 'label' in elem:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
                
            tmp = cv2.resize(tmp, dsize=(new_w, new_h), interpolation=flagval)

            sample[elem] = tmp

        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):

        if random.random() < self.p:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                if elem == 'curr_img' or elem == 'curr_label':
                    new_tmp = []
                    for tmp_ in sample[elem]:
                        tmp_ = cv2.flip(tmp_, flipCode=1)
                        new_tmp.append(tmp_)
                    sample[elem] = new_tmp
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

        return sample

class RandomGaussianBlur(object):

    def __init__(self, prob=0.2):
        self.p = prob

    def __call__(self, sample):

        
        for elem in sample.keys():
            if 'meta' in elem or 'label' in elem:
                continue
            
            if elem == 'curr_img':
                new_tmp = []
                for tmp_ in sample[elem]:
                    if random.random() < self.p:
                        std = random.random() * 1.9 + 0.1  # [0.1, 2]
                        tmp_ = cv2.GaussianBlur(tmp_, (9, 9), sigmaX=std, sigmaY=std)
                    new_tmp.append(tmp_)
                sample[elem] = new_tmp
            else:
                tmp = sample[elem]
                if random.random() < self.p:
                    std = random.random() * 1.9 + 0.1  # [0.1, 2]
                    tmp = cv2.GaussianBlur(tmp, (9, 9), sigmaX=std, sigmaY=std)
                sample[elem] = tmp

        return sample

class SubtractMeanImage(object):
    def __init__(self, mean, change_channels=False):
        self.mean = mean
        self.change_channels = change_channels

    def __call__(self, sample):
        for elem in sample.keys():
            if 'image' in elem:
                if self.change_channels:
                    sample[elem] = sample[elem][:, :, [2, 1, 0]]
                sample[elem] = np.subtract(
                    sample[elem], np.array(self.mean, dtype=np.float32))
        return sample

    def __str__(self):
        return 'SubtractMeanImage' + str(self.mean)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if elem == 'curr_img' or elem == 'curr_label':
                new_tmp = []
                for tmp_ in tmp:
                    if tmp_.ndim == 2:
                        tmp_ = tmp_[:, :, np.newaxis]
                    else:
                        tmp_ = tmp_ / 255.
                        tmp_ -= (0.485, 0.456, 0.406)
                        tmp_ /= (0.229, 0.224, 0.225)
                    tmp_ = tmp_.transpose((2, 0, 1))
                    new_tmp.append(torch.from_numpy(tmp_))
                tmp = new_tmp
            else:
                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]
                else:
                    tmp = tmp / 255.
                    tmp -= (0.485, 0.456, 0.406)
                    tmp /= (0.229, 0.224, 0.225)
                tmp = tmp.transpose((2, 0, 1))
                tmp = torch.from_numpy(tmp)
            sample[elem] = tmp

        return sample

class MultiRestrictSize(object):
    def __init__(self, min_size=None, max_size=800, flip=False, multi_scale=[1.3]):
        self.min_size = min_size
        self.max_size = max_size
        self.multi_scale = multi_scale
        self.flip = flip
        assert ((min_size is None)) or ((max_size is None))

    def __call__(self, sample):
        samples = []
        image = sample['current_img']
        h, w = image.shape[:2]
        for scale in self.multi_scale:
            # Fixed range of scales
            sc = None
            # Align short edge
            if not (self.min_size is None):
                if h > w:
                    short_edge = w
                else:
                    short_edge = h
                if short_edge > self.min_size:
                    sc = float(self.min_size) / short_edge
            else:
                if h > w:
                    long_edge = h
                else:
                    long_edge = w
                if long_edge > self.max_size:
                    sc = float(self.max_size) / long_edge

            if sc is None:
                new_h = h
                new_w = w
            else:
                new_h = sc * h
                new_w = sc * w
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)

            if (new_h - 1) % 16 != 0:
                new_h = int(np.around((new_h - 1) / 16.) * 16 + 1)
            if (new_w - 1) % 16 != 0:
                new_w = int(np.around((new_w - 1) / 16.) * 16 + 1)
                
            if new_h == h and new_w == w:
                samples.append(sample)
            else:
                new_sample = {}
                for elem in sample.keys():
                    if 'meta' in elem:
                        new_sample[elem] = sample[elem]
                        continue
                    tmp = sample[elem]
                    if 'label' in elem:
                        new_sample[elem] = sample[elem]
                        continue
                    else:
                        flagval = cv2.INTER_CUBIC
                        tmp = cv2.resize(tmp, dsize=(new_w, new_h), interpolation=flagval)
                        new_sample[elem] = tmp
                samples.append(new_sample)

            if self.flip:
                now_sample = samples[-1]
                new_sample = {}
                for elem in now_sample.keys():
                    if 'meta' in elem:
                        new_sample[elem] = now_sample[elem].copy()
                        new_sample[elem]['flip'] = True
                        continue
                    tmp = now_sample[elem]
                    tmp = tmp[:, ::-1].copy()
                    new_sample[elem] = tmp
                samples.append(new_sample)

        return samples

class MultiToTensor(object):
    def __call__(self, samples):
        for idx in range(len(samples)):
            sample = samples[idx]
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                if tmp is None:
                    continue

                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]
                else:
                    tmp = tmp / 255.
                    tmp -= (0.485, 0.456, 0.406)
                    tmp /= (0.229, 0.224, 0.225)

                tmp = tmp.transpose((2, 0, 1))
                samples[idx][elem] = torch.from_numpy(tmp)

        return samples

        