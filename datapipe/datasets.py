import random
import numpy as np
from pathlib import Path

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset
from albumentations import SmallestMaxSize

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .degradation_bsrgan.bsrgan_light import degradation_bsrgan_variant, degradation_bsrgan

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import img2tensor
import os
import glob
import math



def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'face':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_crop_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(),
            thv.transforms.ToTensor(),
            thv.transforms.RandomCrop(
                crop_size=kwargs.get('crop_size', None),
                pad_if_needed=True,
                padding_mode='reflect',
                ),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'folder':
        dataset = BaseDataFolder(**dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = BicubicData(**dataset_config['params'])
    elif dataset_config['type'] == 'tempoSpatial':
        dataset = tempoSpatialEval_BicubicData(dataset_config['params'])
    elif dataset_config['type'] == 'bsrgan':
        dataset = BSRGANLightDeg(**dataset_config['params'])
    elif dataset_config['type'] == 'bsrganimagenet':
        dataset = BSRGANLightDegImageNet(**dataset_config['params'])
    elif dataset_config['type'] == 'txt':
        dataset = BaseDataTxt(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
        #dataset = tempoSpatialEval_ExperimentsData(dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseDataFolder(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs=None,
            dir_path_extra=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super(BaseDataFolder, self).__init__()

        file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.dir_path_extra = dir_path_extra
        self.transform = get_transforms(transform_type, transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)
        out_dict = {'image':im, 'lq':im}

        if self.dir_path_extra is not None:
            im_path_extra = Path(self.dir_path_extra) / Path(im_path).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.transform(im_extra)
            out_dict['gt'] = im_extra

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BaseDataTxt(Dataset):
    def __init__(
            self,
            txt_file_path,
            transform_type,
            transform_kwargs=None,
            length=None,
            need_path=False,
            ):
        '''
        transform_kwargs: dict, parameters for transform
        '''
        super().__init__()
        file_paths_all = util_common.readline_txt(txt_file_path)
        self.file_paths_all = file_paths_all

        if length is None:
            self.length = len(file_paths_all)
            self.file_paths = file_paths_all
        else:
            self.length = length
            self.file_paths = random.sample(file_paths_all, length)

        self.transform = get_transforms(transform_type, transform_kwargs)
        self.need_path = need_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='uint8')
        im = self.transform(im)
        out_dict = {'image':im, }

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BSRGANLightDegImageNet(Dataset):
    def __init__(self,
                 dir_paths=None,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=True,
                 degradation='bsrgan_light',
                 use_sharp=False,
                 rescale_gt=True,
                 ):
        super().__init__()
        file_paths_all = []
        if dir_paths is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_paths, im_exts, recursive))
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.mean = mean
        self.std = std
        self.rescale_gt = rescale_gt
        if rescale_gt:
            self.smallest_rescaler = SmallestMaxSize(max_size=gt_size)

        self.gt_size = gt_size
        self.LR_size = int(gt_size / sf)

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=sf, use_sharp=use_sharp)
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_bsrgan_variant, sf=sf, use_sharp=use_sharp)
        else:
            raise ValueError(f'Except bsrgan or bsrgan_light for degradation, now is {degradation}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        h, w = im_hq.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            pad_h = max(0, self.gt_size - h)
            pad_w = max(0, self.gt_size - w)
            im_hq = cv2.copyMakeBorder(im_hq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if self.rescale_gt:
            im_hq = self.smallest_rescaler(image=im_hq)['image']

        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        im_lq, im_hq = self.degradation_process(image=im_hq)
        im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class BSRGANLightDeg(Dataset):
    def __init__(self,
                 dir_paths,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=False,
                 resize_back=False,
                 use_sharp=False,
                 ):
        super().__init__()
        file_paths_all = util_common.scan_files_from_folder(dir_paths, im_exts, recursive)
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.resize_back = resize_back

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.gt_size = gt_size
        self.mean = mean
        self.std = std
        self.use_sharp=use_sharp

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        # random crop
        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        # degradation
        im_lq, im_hq = degradation_bsrgan_variant(im_hq, self.sf, use_sharp=self.use_sharp)
        if self.resize_back:
            im_lq = cv2.resize(im_lq, dsize=(self.gt_size,)*2, interpolation=cv2.INTER_CUBIC)
            im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class BicubicData(Dataset):
    def __init__(
            self,
            sf,
            dir_path=None,
            txt_file_path=None,
            mean=0.5,
            std=0.5,
            hflip=False,
            rotation=False,
            resize_back=False,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            use_sharp=False,
            rescale_gt=True,
            gt_size=256,
            matlab_mode=True,
            ):
        if txt_file_path is None:
            assert dir_path is not None
            file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        else:
            file_paths_all = util_common.readline_txt(txt_file_path)
        self.file_paths_all = file_paths_all

        if length is None:
            self.file_paths = file_paths_all
        else:
            assert len(file_paths_all) >= length
            self.file_paths = random.sample(file_paths_all, length)

        self.sf = sf
        self.mean = mean
        self.std = std
        self.hflip = hflip
        self.rotation = rotation
        self.length = length
        self.need_path = need_path
        self.resize_back = resize_back
        self.use_sharp = use_sharp
        self.rescale_gt = rescale_gt
        self.gt_size = gt_size
        self.matlab_mode = matlab_mode

        self.transform = get_transforms('default', {'mean': mean, 'std': std})
        if rescale_gt:
            self.smallest_rescaler = SmallestMaxSize(max_size=gt_size)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')

        h, w = im_gt.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            pad_h = max(0, self.gt_size - h)
            pad_w = max(0, self.gt_size - w)
            im_gt = cv2.copyMakeBorder(im_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if self.rescale_gt:
            im_gt = self.smallest_rescaler(image=im_gt)['image']

        im_gt = util_image.random_crop(im_gt, self.gt_size)

        im_gt = augment(im_gt, hflip=self.hflip, rotation=self.rotation, return_status=False)

        # imresize
        if self.matlab_mode:
            im_lq = util_image.imresize_np(im_gt, scale=1/self.sf)
        else:
            im_lq = cv2.resize(im_gt, dsize=None, fx=1/self.sf, fy=1/self.sf, interpolation=cv2.INTER_CUBIC)
        if self.resize_back:
            if self.matlab_mode:
                im_lq = util_image.imresize_np(im_gt, scale=self.sf)
            else:
                im_lq = cv2.resize(im_lq, dsize=None, fx=self.sf, fy=self.sf, interpolation=cv2.INTER_CUBIC)
        im_lq = np.clip(im_lq, 0.0, 1.0)

        out = {'lq':self.transform(im_lq), 'gt':self.transform(im_gt)}
        if self.need_path:
            out['path'] = im_path

        return out

class tempoSpatialEval_BicubicData(Dataset):
    def __init__(
            self,
            opt
            ):
        
        
        
        self.opt = opt
        self.dataset = RealESRGANDataset(opt = opt, mode = 'testing')
        self.paths = self.dataset.paths
        self.paths_idx_to_key = self.dataset.paths_idx_to_key
        self.temporal_SR = 8
        self.temp_frame_count = 30
        self.interp_frames = self.temporal_SR - 1 # 7
        self.chunks_in_video = self.temp_frame_count // self.temporal_SR # 3
        self.samples_in_video = self.chunks_in_video * self.interp_frames # 21
        self.total_sample_count = self.samples_in_video * len(self.paths)

        self.sf = opt['sf']
        self.resize_back = opt['resize_back']
        self.rescale_gt = opt['rescale_gt']
        self.gt_size = opt['gt_size']
        self.matlab_mode = opt['matlab_mode']
        self.transform_type = opt['transform_type']
        self.transform_kwargs = opt['transform_kwargs']

        self.transform = get_transforms(self.transform_type, self.transform_kwargs)
        if self.rescale_gt:
            self.smallest_rescaler = SmallestMaxSize(max_size=self.gt_size)

    def __len__(self):
        return int(self.total_sample_count)

    def __getitem__(self, index):
        #im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
        video_idx = index // self.samples_in_video
        video_path = self.paths[video_idx]
        assert len(video_path) == self.temp_frame_count, f"{self.paths_idx_to_key[index]} has {len(video_path)}, but expecting {self.temp_frame_count}"
        frame_id = index % self.samples_in_video
        frame_bucket_id = frame_id // self.interp_frames
        start_time_step = (frame_bucket_id * self.interp_frames) + frame_bucket_id
        end_time_step = start_time_step + self.temporal_SR
        mid_time_idx = (frame_id % self.interp_frames) + 1
        mid_frame_idx = start_time_step + mid_time_idx

        start_path, end_path = (video_path[start_time_step], video_path[end_time_step])
        root_path = self.opt['root_path']
        start_path = f'{root_path}/{start_path}'
        end_path = f'{root_path}/{end_path}'
        start_frame, end_frame = self.dataset.read_scale_dcm(start_path), self.dataset.read_scale_dcm(end_path)

        if (start_time_step + 1) == mid_frame_idx: 
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx, mid_time_idx+1], num_interpolations = self.temporal_SR - 1)
            interp_frames = [start_frame.copy()] + interp_frames
            left = start_frame
            mid = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
            right = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx + 1]}') 
        elif (end_time_step - 1) == mid_frame_idx:
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx], num_interpolations = self.temporal_SR - 1)
            interp_frames = interp_frames + [end_frame.copy()]
            left = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx - 1]}') 
            mid = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
            right = end_frame
        else:
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx, mid_time_idx+1], num_interpolations = self.temporal_SR - 1)
            left = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx - 1]}') 
            mid = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
            right = self.dataset.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx + 1]}') 
            
        interp_image = np.stack(interp_frames, axis = -1) # use this as spaital GT, apply spatial SR on this image, spatially degrade this GT
        GT_image = np.stack([left, mid, right], axis = -1) # use this for final loss calculation as GT, model should output this as GT
        
        im_gt_6ch = np.concatenate([interp_image, GT_image], axis = -1)  # HxWxC with C = 6
        im_gt_6ch = im_gt_6ch.astype(np.float32) / 255 # [0 - 1]
        h, w = im_gt_6ch.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            pad_h = max(0, self.gt_size - h)
            pad_w = max(0, self.gt_size - w)
            im_gt = cv2.copyMakeBorder(im_gt_6ch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if self.rescale_gt:
            im_gt_6ch = self.smallest_rescaler(image=im_gt_6ch)['image']

        im_gt_6ch = util_image.random_crop(im_gt_6ch, self.gt_size)
        im_gt, im_gt_golden = im_gt_6ch[:, :, 0:3], im_gt_6ch[:, :, 3:6]

        # imresize
        if self.matlab_mode:
            im_lq = util_image.imresize_np(im_gt, scale=1/self.sf)
        else:
            im_lq = cv2.resize(im_gt, dsize=None, fx=1/self.sf, fy=1/self.sf, interpolation=cv2.INTER_CUBIC)
        if self.resize_back:
            if self.matlab_mode:
                im_lq = util_image.imresize_np(im_gt, scale=self.sf)
            else:
                im_lq = cv2.resize(im_lq, dsize=None, fx=self.sf, fy=self.sf, interpolation=cv2.INTER_CUBIC)
        im_lq = np.clip(im_lq, 0.0, 1.0)

        out = {'lq':self.transform(im_lq), 'gt':self.transform(im_gt_golden)} # [0, 1] to [-1, 1]

        return out


class tempoSpatialEval_ExperimentsData(Dataset):
    def __init__(
            self,
            opt,
            max_batch_size = 16,
            dataset_path = f"second-annual-data-science-bowl-v2/validate/validate",
            pat_to_test = 205
            ):
        
        
        
        self.opt = opt
        self.max_batch_size = max_batch_size
        self.root_path = self.opt['root_path']
        self.dataset_path = f"{self.root_path}/{dataset_path}"
        self.dataset = RealESRGANDataset(opt = opt, mode = 'training')
        self.temporal_SR = 8
        self.temp_frame_count = 30
        
        self.interp_frames = self.temporal_SR - 1 # 7
        self.chunks_in_video = self.temp_frame_count // self.temporal_SR # 3
        self.samples_in_video = self.chunks_in_video * self.interp_frames # 21
        self.paths = [] #self.max_batch_size * number_of_pat
        patients_name = sorted(os.listdir(self.dataset_path))
        pat_to_test = min(pat_to_test, len(patients_name))
        patients_name = patients_name[:pat_to_test]
        for pat_name in patients_name:
            print(f"{pat_name}")
            pat_folder = f"{self.dataset_path}/{pat_name}"
            slices_name = sorted(os.listdir(f"{pat_folder}/study"))
            for slice_name in slices_name:
                if "sax_" not in slice_name: continue
                slice_folder = f"{pat_folder}/study/{slice_name}"
                slice_time_li = sorted(glob.glob(f"{slice_folder}/*.dcm"))
                if len(slice_time_li) != self.temp_frame_count: continue
                self.paths.append(slice_time_li)
        
        
        self.total_sample_count = self.samples_in_video * len(self.paths)

        self.sf = 4
        self.rescale_gt = opt['rescale_gt']
        self.gt_size = opt['gt_size']

        
        if self.rescale_gt:
            self.smallest_rescaler = SmallestMaxSize(max_size=self.gt_size)


    def get_degradation_kernels(self, ):
         # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.dataset.kernel_range1)
        if np.random.uniform() < self.dataset.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.dataset.kernel_list,
                self.dataset.kernel_prob,
                kernel_size,
                self.dataset.blur_sigma,
                self.dataset.blur_sigma, [-math.pi, math.pi],
                self.dataset.betag_range,
                self.dataset.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.dataset.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))


         # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.dataset.opt['final_sinc_prob']:
            kernel_size = random.choice(self.dataset.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.dataset.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.dataset.pulse_tensor
        kernel = torch.FloatTensor(kernel)

        return kernel, sinc_kernel


    def __len__(self):
        return int(self.total_sample_count)

    def __getitem__(self, index):
        #im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
        video_idx = index // self.samples_in_video
        video_path = self.paths[video_idx]
        assert len(video_path) == self.temp_frame_count, f"It has {len(video_path)}, but expecting {self.temp_frame_count}"
        frame_id = index % self.samples_in_video
        frame_bucket_id = frame_id // self.interp_frames
        start_time_step = (frame_bucket_id * self.interp_frames) + frame_bucket_id
        end_time_step = start_time_step + self.temporal_SR
        mid_time_idx = (frame_id % self.interp_frames) + 1
        mid_frame_idx = start_time_step + mid_time_idx

        start_path, end_path = (video_path[start_time_step], video_path[end_time_step])
        
        start_frame, end_frame = self.dataset.read_scale_dcm(start_path), self.dataset.read_scale_dcm(end_path)

        if (start_time_step + 1) == mid_frame_idx: 
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx, mid_time_idx+1], num_interpolations = self.temporal_SR - 1)
            interp_frames = [start_frame.copy()] + interp_frames
            left = start_frame
            mid = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx]}') 
            right = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx + 1]}') 
        elif (end_time_step - 1) == mid_frame_idx:
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx], num_interpolations = self.temporal_SR - 1)
            interp_frames = interp_frames + [end_frame.copy()]
            left = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx - 1]}') 
            mid = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx]}') 
            right = end_frame
        else:
            interp_frames = self.dataset.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx, mid_time_idx+1], num_interpolations = self.temporal_SR - 1)
            left = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx - 1]}') 
            mid = self.dataset.read_scale_dcm(f'{video_path[mid_frame_idx]}') 
            right = self.dataset.read_scale_dcm(f'/{video_path[mid_frame_idx + 1]}') 
            
        interp_image = np.stack(interp_frames, axis = -1) # use this as spaital GT, apply spatial SR on this image, spatially degrade this GT
        GT_image = np.stack([left, mid, right], axis = -1) # use this for final loss calculation as GT, model should output this as GT
        
        im_gt_6ch = np.concatenate([interp_image, GT_image], axis = -1)  # HxWxC with C = 6
        im_gt_6ch = (im_gt_6ch.astype(np.float32) / 255).clip(0.0, 1.0) # [0 - 1]
        # BGR to RGB, HWC to CHW, numpy to tensor
        im_gt_6ch = img2tensor([im_gt_6ch], bgr2rgb=True, float32=True)[0] #CxHxW with C = 6 [0 - 1]
        kernel, sinc_kernel = self.get_degradation_kernels()


        

        out = {'gt': im_gt_6ch, 'kernel1': kernel, 'sinc_kernel': sinc_kernel} # [0, 1]

        return out
